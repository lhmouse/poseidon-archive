// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "connection.hpp"
#include "thread_context.hpp"
#include "exception.hpp"
#include "formatting.hpp"
#include "../raii.hpp"
#include "../log.hpp"
#include "../time.hpp"
#include "../system_exception.hpp"
#include "../uuid.hpp"
#include <cstdlib>
#include <mysql.h>

namespace Poseidon {
namespace MySql {

namespace {
	struct Closer {
		CONSTEXPR ::MYSQL *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::MYSQL *mysql) const NOEXCEPT {
			::mysql_close(mysql);
		}
	};
	struct ResultDeleter {
		CONSTEXPR ::MYSQL_RES *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::MYSQL_RES *result) const NOEXCEPT {
			::mysql_free_result(result);
		}
	};

	struct FieldComparator {
		bool operator()(const char *lhs, const char *rhs) const NOEXCEPT {
			return std::strcmp(lhs, rhs) < 0;
		}
	};

	class DelegatedConnection : public Connection {
	private:
		const ThreadContext m_context;
		const SharedNts m_schema;

		::MYSQL m_mysql_storage;
		UniqueHandle<Closer> m_mysql;

		UniqueHandle<ResultDeleter> m_result;
		boost::container::flat_map<const char *, std::size_t, FieldComparator> m_fields;
		::MYSQL_ROW m_row;
		unsigned long *m_lengths;

	public:
		DelegatedConnection(const char *server_addr, boost::uint16_t server_port, const char *user_name, const char *password, const char *schema, bool use_ssl, const char *charset)
			: m_schema(schema)
			, m_row(NULLPTR), m_lengths(NULLPTR)
		{
			DEBUG_THROW_UNLESS(m_mysql.reset(::mysql_init(&m_mysql_storage)), BasicException, sslit("::mysql_init() failed"));
			DEBUG_THROW_UNLESS(::mysql_options(m_mysql.get(), MYSQL_OPT_COMPRESS, NULLPTR) == 0, BasicException, sslit("::mysql_options() failed, trying to set MYSQL_OPT_COMPRESS"));
			static CONSTEXPR const ::my_bool TRUE_VALUE = true;
			DEBUG_THROW_UNLESS(::mysql_options(m_mysql.get(), MYSQL_OPT_RECONNECT, &TRUE_VALUE) == 0, BasicException, sslit("::mysql_options() failed, trying to set MYSQL_OPT_RECONNECT"));
			DEBUG_THROW_UNLESS(::mysql_options(m_mysql.get(), MYSQL_SET_CHARSET_NAME, charset) == 0, BasicException, sslit("::mysql_options() failed, trying to set MYSQL_OPT_RECONNECT"));
			unsigned long flags = 0;
			if(use_ssl){
				flags |= CLIENT_SSL;
			}
			DEBUG_THROW_UNLESS(::mysql_real_connect(m_mysql.get(), server_addr, user_name, password, schema, server_port, NULLPTR, flags), Exception, m_schema, ::mysql_errno(m_mysql.get()), SharedNts(::mysql_error(m_mysql.get())));
		}

	private:
		bool find_field_and_check(const char *&data, std::size_t &size, const char *name) const {
			if(!m_row){
				LOG_POSEIDON_WARNING("No more results available.");
				return false;
			}
			const AUTO(it, m_fields.find(name));
			if(it == m_fields.end()){
				LOG_POSEIDON_WARNING("Field not found: name = ", name);
				return false;
			}
			data = m_row[it->second];
			if(!data){
				LOG_POSEIDON_DEBUG("Field is null: name = ", name);
				return false;
			}
			size = m_lengths[it->second];
			return true;
		}

	public:
		void do_execute_sql(const char *sql, std::size_t len){
			do_discard_result();

			DEBUG_THROW_UNLESS(::mysql_real_query(m_mysql.get(), sql, len) == 0, Exception, m_schema, ::mysql_errno(m_mysql.get()), SharedNts(::mysql_error(m_mysql.get())));

			if(!m_result.reset(::mysql_use_result(m_mysql.get()))){
				DEBUG_THROW_UNLESS(::mysql_errno(m_mysql.get()) == 0, Exception, m_schema, ::mysql_errno(m_mysql.get()), SharedNts(::mysql_error(m_mysql.get())));
				// 没有返回结果。
			} else {
				const AUTO(fields, ::mysql_fetch_fields(m_result.get()));
				const AUTO(count, ::mysql_num_fields(m_result.get()));
				m_fields.reserve(count);
				for(std::size_t i = 0; i < count; ++i){
					const char *const name = fields[i].name;
					DEBUG_THROW_UNLESS(m_fields.insert(std::make_pair(name, i)).second, BasicException, sslit("Duplicate field"));
					LOG_POSEIDON_TRACE("MySQL result field: name = ", name, ", index = ", i);
				}
			}
		}
		void do_discard_result() NOEXCEPT {
			m_result.reset();
			m_fields.clear();
			m_row = NULLPTR;
			m_lengths = NULLPTR;
		}

		boost::uint64_t do_get_insert_id() const {
			return ::mysql_insert_id(m_mysql.get());
		}

		bool do_fetch_row(){
			if(m_fields.empty()){
				LOG_POSEIDON_DEBUG("Empty set returned from MySQL server.");
				return false;
			}

			const AUTO(row, ::mysql_fetch_row(m_result.get()));
			if(!row){
				LOG_POSEIDON_DEBUG("No more data.");
				return false;
			}
			const AUTO(lengths, ::mysql_fetch_lengths(m_result.get()));
			m_row = row;
			m_lengths = lengths;
			return true;
		}

		boost::int64_t do_get_signed(const char *name) const {
			const char *data;
			std::size_t size;
			if(!find_field_and_check(data, size, name)){
				return VAL_INIT;
			}
			char *eptr;
			const AUTO(val, ::strtoll(data, &eptr, 10));
			DEBUG_THROW_UNLESS(*eptr == 0, BasicException, sslit("Could not convert field data to long long"));
			return val;
		}
		boost::uint64_t do_get_unsigned(const char *name) const {
			const char *data;
			std::size_t size;
			if(!find_field_and_check(data, size, name)){
				return VAL_INIT;
			}
			char *eptr;
			const AUTO(val, ::strtoull(data, &eptr, 10));
			DEBUG_THROW_UNLESS(*eptr == 0, BasicException, sslit("Could not convert field data to unsigned long long"));
			return val;
		}
		double do_get_double(const char *name) const {
			const char *data;
			std::size_t size;
			if(!find_field_and_check(data, size, name)){
				return VAL_INIT;
			}
			char *eptr;
			const AUTO(val, ::strtod(data, &eptr));
			DEBUG_THROW_UNLESS(*eptr == 0, BasicException, sslit("Could not convert field data to double"));
			return val;
		}
		std::string do_get_string(const char *name) const {
			const char *data;
			std::size_t size;
			if(!find_field_and_check(data, size, name)){
				return VAL_INIT;
			}
			return std::string(data, size);
		}
		boost::uint64_t do_get_datetime(const char *name) const {
			const char *data;
			std::size_t size;
			if(!find_field_and_check(data, size, name)){
				return VAL_INIT;
			}
			return scan_time(data);
		}
		Uuid do_get_uuid(const char *name) const {
			const char *data;
			std::size_t size;
			if(!find_field_and_check(data, size, name)){
				return VAL_INIT;
			}
			DEBUG_THROW_UNLESS(size == 36, BasicException, sslit("Invalid UUID string"));
			return Uuid(reinterpret_cast<const char (&)[36]>(data[0]));
		}
		std::basic_string<unsigned char> do_get_blob(const char *name) const {
			const char *data;
			std::size_t size;
			if(!find_field_and_check(data, size, name)){
				return VAL_INIT;
			}
			return std::basic_string<unsigned char>(reinterpret_cast<const unsigned char *>(data), size);
		}
	};
}

boost::shared_ptr<Connection> Connection::create(const char *server_addr, boost::uint16_t server_port, const char *user_name, const char *password, const char *schema, bool use_ssl, const char *charset){
	return boost::make_shared<DelegatedConnection>(server_addr, server_port, user_name, password, schema, use_ssl, charset);
}

Connection::~Connection(){ }

void Connection::execute_sql(const char *sql, std::size_t len){
	static_cast<DelegatedConnection &>(*this).do_execute_sql(sql, len);
}
void Connection::discard_result() NOEXCEPT {
	static_cast<DelegatedConnection &>(*this).do_discard_result();
}

boost::uint64_t Connection::get_insert_id() const {
	return static_cast<const DelegatedConnection &>(*this).do_get_insert_id();
}
bool Connection::fetch_row(){
	return static_cast<DelegatedConnection &>(*this).do_fetch_row();
}

boost::int64_t Connection::get_signed(const char *name) const {
	return static_cast<const DelegatedConnection &>(*this).do_get_signed(name);
}
boost::uint64_t Connection::get_unsigned(const char *name) const {
	return static_cast<const DelegatedConnection &>(*this).do_get_unsigned(name);
}
double Connection::get_double(const char *name) const {
	return static_cast<const DelegatedConnection &>(*this).do_get_double(name);
}
std::string Connection::get_string(const char *name) const {
	return static_cast<const DelegatedConnection &>(*this).do_get_string(name);
}
boost::uint64_t Connection::get_datetime(const char *name) const {
	return static_cast<const DelegatedConnection &>(*this).do_get_datetime(name);
}
Uuid Connection::get_uuid(const char *name) const {
	return static_cast<const DelegatedConnection &>(*this).do_get_uuid(name);
}
std::basic_string<unsigned char> Connection::get_blob(const char *name) const {
	return static_cast<const DelegatedConnection &>(*this).do_get_blob(name);
}

}
}
