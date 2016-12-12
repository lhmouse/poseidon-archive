// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "connection.hpp"
#include "thread_context.hpp"
#include "exception.hpp"
#include "formatting.hpp"
#include <boost/container/flat_map.hpp>
#include <string.h>
#include <stdlib.h>
#include <mysql.h>
#include "../raii.hpp"
#include "../log.hpp"
#include "../time.hpp"
#include "../system_exception.hpp"
#include "../uuid.hpp"

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

		struct ColumnComparator {
			bool operator()(const char *lhs, const char *rhs) const NOEXCEPT {
				return std::strcmp(lhs, rhs) < 0;
			}
		};

#define DEBUG_THROW_MYSQL_EXCEPTION(mysql_, schema_)	\
	DEBUG_THROW(::Poseidon::MySql::Exception, schema_, ::mysql_errno(mysql_), ::Poseidon::SharedNts(::mysql_error(mysql_)))

		class DelegatedConnection : public Connection {
		private:
			const ThreadContext m_context;
			const SharedNts m_schema;

			::MYSQL m_mysql_object;
			UniqueHandle<Closer> m_mysql;

			UniqueHandle<ResultDeleter> m_result;
			boost::container::flat_map<const char *, std::size_t, ColumnComparator> m_columns;

			::MYSQL_ROW m_row;
			unsigned long *m_lengths;

		public:
			DelegatedConnection(const char *server_addr, unsigned server_port,
				const char *user_name, const char *password, const char *schema, bool use_ssl, const char *charset)
				: m_schema(schema)
				, m_row(NULLPTR), m_lengths(NULLPTR)
			{
				if(!m_mysql.reset(::mysql_init(&m_mysql_object))){
					DEBUG_THROW(SystemException, ENOMEM);
				}

				if(::mysql_options(m_mysql.get(), MYSQL_OPT_COMPRESS, NULLPTR) != 0){
					DEBUG_THROW_MYSQL_EXCEPTION(m_mysql.get(), m_schema);
				}
				const ::my_bool TRUE_VALUE = true;
				if(::mysql_options(m_mysql.get(), MYSQL_OPT_RECONNECT, &TRUE_VALUE) != 0){
					DEBUG_THROW_MYSQL_EXCEPTION(m_mysql.get(), m_schema);
				}
				if(::mysql_options(m_mysql.get(), MYSQL_SET_CHARSET_NAME, charset) != 0){
					DEBUG_THROW_MYSQL_EXCEPTION(m_mysql.get(), m_schema);
				}

				if(!::mysql_real_connect(m_mysql.get(), server_addr, user_name,
					password, schema, server_port, NULLPTR, use_ssl ? CLIENT_SSL : 0))
				{
					DEBUG_THROW_MYSQL_EXCEPTION(m_mysql.get(), m_schema);
				}
			}

		public:
			void do_execute_sql(const char *sql, std::size_t len){
				do_discard_result();

				if(::mysql_real_query(m_mysql.get(), sql, len) != 0){
					DEBUG_THROW_MYSQL_EXCEPTION(m_mysql.get(), m_schema);
				}

				if(!m_result.reset(::mysql_use_result(m_mysql.get()))){
					if(::mysql_errno(m_mysql.get()) != 0){
						DEBUG_THROW_MYSQL_EXCEPTION(m_mysql.get(), m_schema);
					}
					// 没有返回结果。
				} else {
					const AUTO(fields, ::mysql_fetch_fields(m_result.get()));
					const AUTO(count, ::mysql_num_fields(m_result.get()));
					m_columns.reserve(count);
					for(std::size_t i = 0; i < count; ++i){
						const char *const name = fields[i].name;
						if(!m_columns.insert(std::make_pair(name, i)).second){
							LOG_POSEIDON_ERROR("Duplicate column in MySQL result set: ", name);
							DEBUG_THROW(BasicException, sslit("Duplicate column"));
						}
						LOG_POSEIDON_TRACE("MySQL result column: name = ", name, ", index = ", i);
					}
				}
			}
			void do_discard_result() NOEXCEPT {
				m_result.reset();
				m_columns.clear();
				m_row = NULLPTR;
			}

			boost::uint64_t do_get_insert_id() const {
				return ::mysql_insert_id(m_mysql.get());
			}

			bool do_fetch_row(){
				if(m_columns.empty()){
					LOG_POSEIDON_DEBUG("Empty set returned from MySQL server.");
					return false;
				}
				m_row = ::mysql_fetch_row(m_result.get());
				if(!m_row){
					if(::mysql_errno(m_mysql.get()) != 0){
						DEBUG_THROW_MYSQL_EXCEPTION(m_mysql.get(), m_schema);
					}
					return false;
				}
				m_lengths = ::mysql_fetch_lengths(m_result.get());
				return true;
			}

			boost::int64_t do_get_signed(const char *column) const {
				const AUTO(it, m_columns.find(column));
				if(it == m_columns.end()){
					LOG_POSEIDON_ERROR("Column not found: ", column);
					DEBUG_THROW(BasicException, sslit("Column not found"));
				}
				const AUTO(data, m_row[it->second]);
				if(!data || (data[0] == 0)){
					return 0;
				}
				char *endptr;
				const AUTO(val, ::strtoll(data, &endptr, 10));
				if(endptr[0] != 0){
					LOG_POSEIDON_ERROR("Could not convert column data to long long: ", data);
					DEBUG_THROW(BasicException, sslit("Invalid data format"));
				}
				return val;
			}
			boost::uint64_t do_get_unsigned(const char *column) const {
				const AUTO(it, m_columns.find(column));
				if(it == m_columns.end()){
					LOG_POSEIDON_ERROR("Column not found: ", column);
					DEBUG_THROW(BasicException, sslit("Column not found"));
				}
				const AUTO(data, m_row[it->second]);
				if(!data || (data[0] == 0)){
					return 0;
				}
				char *endptr;
				const AUTO(val, ::strtoull(data, &endptr, 10));
				if(endptr[0] != 0){
					LOG_POSEIDON_ERROR("Could not convert column data to unsigned long long: ", data);
					DEBUG_THROW(BasicException, sslit("Invalid data format"));
				}
				return val;
			}
			double do_get_double(const char *column) const {
				const AUTO(it, m_columns.find(column));
				if(it == m_columns.end()){
					LOG_POSEIDON_ERROR("Column not found: ", column);
					DEBUG_THROW(BasicException, sslit("Column not found"));
				}
				const AUTO(data, m_row[it->second]);
				if(!data || (data[0] == 0)){
					return 0;
				}
				char *endptr;
				const AUTO(val, ::strtod(data, &endptr));
				if(endptr[0] != 0){
					LOG_POSEIDON_ERROR("Could not convert column data to double: ", data);
					DEBUG_THROW(BasicException, sslit("Invalid data format"));
				}
				return val;
			}
			std::string do_get_string(const char *column) const {
				const AUTO(it, m_columns.find(column));
				if(it == m_columns.end()){
					LOG_POSEIDON_ERROR("Column not found: ", column);
					DEBUG_THROW(BasicException, sslit("Column not found"));
				}
				std::string val;
				const AUTO(data, m_row[it->second]);
				if(data){
					val.assign(data, m_lengths[it->second]);
				}
				return val;
			}
			boost::uint64_t do_get_datetime(const char *column) const {
				const AUTO(it, m_columns.find(column));
				if(it == m_columns.end()){
					LOG_POSEIDON_ERROR("Column not found: ", column);
					DEBUG_THROW(BasicException, sslit("Column not found"));
				}
				const AUTO(data, m_row[it->second]);
				if(!data || (data[0] == 0)){
					return 0;
				}
				return scan_time(data);
			}
			Uuid do_get_uuid(const char *column) const {
				const AUTO(it, m_columns.find(column));
				if(it == m_columns.end()){
					LOG_POSEIDON_ERROR("Column not found: ", column);
					DEBUG_THROW(BasicException, sslit("Column not found"));
				}
				const AUTO(data, m_row[it->second]);
				if(!data || (data[0] == 0)){
					return Uuid();
				}
				if(std::strlen(data) != 36){
					LOG_POSEIDON_ERROR("Invalid UUID string: ", data);
					DEBUG_THROW(BasicException, sslit("Invalid UUID string"));
				}
				return Uuid(reinterpret_cast<const char (&)[36]>(data[0]));
			}
		};
	}

	boost::shared_ptr<Connection> Connection::create(const char *server_addr, unsigned server_port,
		const char *user_name, const char *password, const char *schema, bool use_ssl, const char *charset)
	{
		return boost::make_shared<DelegatedConnection>(server_addr, server_port,
			user_name, password, schema, use_ssl, charset);
	}

	Connection::~Connection(){
	}

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

	boost::int64_t Connection::get_signed(const char *column) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_signed(column);
	}
	boost::uint64_t Connection::get_unsigned(const char *column) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_unsigned(column);
	}
	double Connection::get_double(const char *column) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_double(column);
	}
	std::string Connection::get_string(const char *column) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_string(column);
	}
	boost::uint64_t Connection::get_datetime(const char *column) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_datetime(column);
	}
	Uuid Connection::get_uuid(const char *column) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_uuid(column);
	}
}

}
