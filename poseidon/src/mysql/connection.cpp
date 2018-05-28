// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "connection.hpp"
#include "exception.hpp"
#include "formatting.hpp"
#include "../raii.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../time.hpp"
#include "../system_exception.hpp"
#include <mysql/mysql.h>

namespace Poseidon {
namespace Mysql {

namespace {
	struct Closer {
		CONSTEXPR ::MYSQL *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::MYSQL *mysql) const NOEXCEPT {
			::mysql_close(mysql);
		}
	};
	struct Result_deleter {
		CONSTEXPR ::MYSQL_RES *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::MYSQL_RES *result) const NOEXCEPT {
			::mysql_free_result(result);
		}
	};

	struct Field_comparator {
		bool operator()(const char *lhs, const char *rhs) const NOEXCEPT {
			return std::strcmp(lhs, rhs) < 0;
		}
	};

	class Delegated_connection FINAL : public Connection {
	private:
		Rcnts m_schema;
		::MYSQL m_mysql_storage;
		Unique_handle<Closer> m_mysql;

		Unique_handle<Result_deleter> m_result;
		boost::container::flat_map<const char *, std::size_t, Field_comparator> m_fields;
		::MYSQL_ROW m_row;
		unsigned long *m_lengths;

	public:
		Delegated_connection(const char *server_addr, boost::uint16_t server_port, const char *user_name, const char *password, const char *schema, bool use_ssl, const char *charset)
			: m_schema(schema)
			, m_row(NULLPTR), m_lengths(NULLPTR)
		{
			POSEIDON_PROFILE_ME;

			POSEIDON_THROW_UNLESS(m_mysql.reset(::mysql_init(&m_mysql_storage)), Basic_exception, Rcnts::view("::mysql_init() failed"));
			POSEIDON_THROW_UNLESS(::mysql_options(m_mysql.get(), MYSQL_OPT_COMPRESS, NULLPTR) == 0, Basic_exception, Rcnts::view("::mysql_options() failed, trying to set MYSQL_OPT_COMPRESS"));
			static CONSTEXPR const ::my_bool s_true_value = true;
			POSEIDON_THROW_UNLESS(::mysql_options(m_mysql.get(), MYSQL_OPT_RECONNECT, &s_true_value) == 0, Basic_exception, Rcnts::view("::mysql_options() failed, trying to set MYSQL_OPT_RECONNECT"));
			POSEIDON_THROW_UNLESS(::mysql_options(m_mysql.get(), MYSQL_SET_CHARSET_NAME, charset) == 0, Basic_exception, Rcnts::view("::mysql_options() failed, trying to set MYSQL_OPT_RECONNECT"));
			unsigned long flags = 0;
			if(use_ssl){
				flags |= CLIENT_SSL;
			}
			POSEIDON_THROW_UNLESS(::mysql_real_connect(m_mysql.get(), server_addr, user_name, password, schema, server_port, NULLPTR, flags), Exception, m_schema, ::mysql_errno(m_mysql.get()), Rcnts(::mysql_error(m_mysql.get())));
		}

	private:
		bool find_field_and_check(const char *&data, std::size_t &size, const char *name) const {
			POSEIDON_PROFILE_ME;

			if(!m_row){
				POSEIDON_LOG_WARNING("No more results available.");
				return false;
			}
			const AUTO(it, m_fields.find(name));
			if(it == m_fields.end()){
				POSEIDON_LOG_WARNING("Field not found: name = ", name);
				return false;
			}
			data = m_row[it->second];
			if(!data){
				POSEIDON_LOG_DEBUG("Field is `null`: name = ", name);
				return false;
			}
			size = m_lengths[it->second];
			return true;
		}

	public:
		void execute_sql_explicit(const char *sql, std::size_t len) OVERRIDE {
			POSEIDON_PROFILE_ME;

			discard_result();

			POSEIDON_LOG_DEBUG("Sending query to MySQL server: ", std::string(sql, len));
			POSEIDON_THROW_UNLESS(::mysql_real_query(m_mysql.get(), sql, len) == 0, Exception, m_schema, ::mysql_errno(m_mysql.get()), Rcnts(::mysql_error(m_mysql.get())));
			POSEIDON_THROW_UNLESS(::mysql_errno(m_mysql.get()) == 0, Exception, m_schema, ::mysql_errno(m_mysql.get()), Rcnts(::mysql_error(m_mysql.get())));
			if(m_result.reset(::mysql_use_result(m_mysql.get()))){
				const AUTO(fields, ::mysql_fetch_fields(m_result.get()));
				const AUTO(count, ::mysql_num_fields(m_result.get()));
				m_fields.reserve(count);
				for(std::size_t i = 0; i < count; ++i){
					const char *const name = fields[i].name;
					POSEIDON_THROW_UNLESS(m_fields.emplace(name, i).second, Basic_exception, Rcnts::view("Duplicate field"));
					POSEIDON_LOG_TRACE("MySQL result field: name = ", name, ", index = ", i);
				}
			} else {
				POSEIDON_LOG_DEBUG("No result was returned from MySQL server.");
			}
		}
		void discard_result() NOEXCEPT OVERRIDE {
			POSEIDON_PROFILE_ME;

			m_result.reset();
			m_fields.clear();
			m_row = NULLPTR;
			m_lengths = NULLPTR;
		}

		boost::uint64_t get_insert_id() const OVERRIDE {
			return ::mysql_insert_id(m_mysql.get());
		}

		bool fetch_row() OVERRIDE {
			POSEIDON_PROFILE_ME;

			if(m_fields.empty()){
				POSEIDON_LOG_DEBUG("Empty set returned from MySQL server.");
				return false;
			}
			const AUTO(row, ::mysql_fetch_row(m_result.get()));
			if(!row){
				POSEIDON_LOG_DEBUG("No more data.");
				return false;
			}
			const AUTO(lengths, ::mysql_fetch_lengths(m_result.get()));
			m_row = row;
			m_lengths = lengths;
			return true;
		}

		bool get_boolean(const char *name) const OVERRIDE {
			POSEIDON_PROFILE_ME;
			POSEIDON_LOG_TRACE("Getting field as `boolean`: ", name);

			bool value = false;
			const char *data;
			std::size_t size;
			if(find_field_and_check(data, size, name)){
				value = (size != 0) && (std::strcmp(data, "0") != 0);
			}
			return value;
		}
		boost::int64_t get_signed(const char *name) const OVERRIDE {
			POSEIDON_PROFILE_ME;
			POSEIDON_LOG_TRACE("Getting field as `signed`: ", name);

			boost::int64_t value = 0;
			const char *data;
			std::size_t size;
			if(find_field_and_check(data, size, name)){
				char *eptr;
				value = ::strtoll(data, &eptr, 0);
				POSEIDON_THROW_UNLESS(*eptr == 0, Basic_exception, Rcnts::view("Could not convert field data to `long long`"));
			}
			return value;
		}
		boost::uint64_t get_unsigned(const char *name) const OVERRIDE {
			POSEIDON_PROFILE_ME;
			POSEIDON_LOG_TRACE("Getting field as `unsigned`: ", name);

			boost::uint64_t value = 0;
			const char *data;
			std::size_t size;
			if(find_field_and_check(data, size, name)){
				char *eptr;
				value = ::strtoull(data, &eptr, 0);
				POSEIDON_THROW_UNLESS(*eptr == 0, Basic_exception, Rcnts::view("Could not convert field data to `unsigned long long`"));
			}
			return value;
		}
		double get_double(const char *name) const OVERRIDE {
			POSEIDON_PROFILE_ME;
			POSEIDON_LOG_TRACE("Getting field as `double`: ", name);

			double value = 0;
			const char *data;
			std::size_t size;
			if(find_field_and_check(data, size, name)){
				char *eptr;
				value = ::strtod(data, &eptr);
				POSEIDON_THROW_UNLESS(*eptr == 0, Basic_exception, Rcnts::view("Could not convert field data to `double`"));
			}
			return value;
		}
		std::string get_string(const char *name) const OVERRIDE {
			POSEIDON_PROFILE_ME;
			POSEIDON_LOG_TRACE("Getting field as `string`: ", name);

			std::string value;
			const char *data;
			std::size_t size;
			if(find_field_and_check(data, size, name)){
				value.assign(data, size);
			}
			return value;
		}
		boost::uint64_t get_datetime(const char *name) const OVERRIDE {
			POSEIDON_PROFILE_ME;
			POSEIDON_LOG_TRACE("Getting field as `datetime`: ", name);

			boost::uint64_t value = 0;
			const char *data;
			std::size_t size;
			if(find_field_and_check(data, size, name)){
				value = scan_time(data);
			}
			return value;
		}
		Uuid get_uuid(const char *name) const OVERRIDE {
			POSEIDON_PROFILE_ME;
			POSEIDON_LOG_TRACE("Getting field as `uuid`: ", name);

			Uuid value;
			const char *data;
			std::size_t size;
			if(find_field_and_check(data, size, name)){
				POSEIDON_THROW_UNLESS(size == 36, Basic_exception, Rcnts::view("Invalid UUID string length"));
				value.from_string(*reinterpret_cast<const char (*)[36]>(data));
			}
			return value;
		}
		Stream_buffer get_blob(const char *name) const OVERRIDE {
			POSEIDON_PROFILE_ME;
			POSEIDON_LOG_TRACE("Getting field as `blob`: ", name);

			Stream_buffer value;
			const char *data;
			std::size_t size;
			if(find_field_and_check(data, size, name)){
				value.put(data, size);
			}
			return value;
		}
	};
}

boost::shared_ptr<Connection> Connection::create(const char *server_addr, boost::uint16_t server_port, const char *user_name, const char *password, const char *schema, bool use_ssl, const char *charset){
	return boost::make_shared<Delegated_connection>(server_addr, server_port, user_name, password, schema, use_ssl, charset);
}

Connection::~Connection(){
	//
}

}
}
