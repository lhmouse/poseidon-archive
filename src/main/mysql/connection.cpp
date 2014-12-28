// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "connection.hpp"
#include "exception.hpp"
#include <string.h>
#include <stdlib.h>
#include <mysql/mysql.h>
#include "../raii.hpp"
#include "../log.hpp"
using namespace Poseidon;

namespace {

struct MySqlCloser {
	CONSTEXPR ::MYSQL *operator()() const NOEXCEPT {
		return NULLPTR;
	}
	void operator()(::MYSQL *mysql) const NOEXCEPT {
		::mysql_close(mysql);
	}
};

struct MySqlResultDeleter {
	CONSTEXPR ::MYSQL_RES *operator()() const NOEXCEPT {
		return NULLPTR;
	}
	void operator()(::MYSQL_RES *result) const NOEXCEPT {
		::mysql_free_result(result);
	}
};

typedef std::vector<std::pair<const char *, std::size_t> > MySqlColumns;

class MySqlConnectionDelegate : public MySqlConnection {

#define THROW_MYSQL_EXCEPTION	\
	DEBUG_THROW(MySqlException,	\
		::mysql_errno(m_mysql.get()), SharedNts(::mysql_error(m_mysql.get())))

private:
	// 若返回 true 则 pos 指向找到的列，否则 pos 指向下一个。
	template<typename IteratorT, typename VectorT>
	static bool lowerBoundColumn(IteratorT &pos, VectorT &columns, const char *name){
		AUTO(lower, columns.begin());
		AUTO(upper, columns.end());
		for(;;){
			if(lower == upper){
				pos = lower;
				return false;
			}
			const AUTO(middle, lower + (upper - lower) / 2);
			const int result = ::strcasecmp(middle->first, name);
			if(result == 0){
				pos = middle;
				return true;
			} else if(result < 0){
				upper = middle;
			} else {
				lower = middle + 1;
			}
		}
	}

private:
	::MYSQL m_mysqlObject;
	UniqueHandle<MySqlCloser> m_mysql;

	UniqueHandle<MySqlResultDeleter> m_result;
	MySqlColumns m_columns;

	::MYSQL_ROW m_row;
	unsigned long *m_lengths;

public:
	MySqlConnectionDelegate(const std::string &serverAddr, unsigned serverPort,
		const std::string &userName, const std::string &password, const std::string &schema,
		bool useSsl, const std::string &charset)
		: m_row(NULLPTR), m_lengths(NULLPTR)
	{
		if(!m_mysql.reset(::mysql_init(&m_mysqlObject))){
			DEBUG_THROW(SystemError, ENOMEM);
		}

		if(::mysql_options(m_mysql.get(), MYSQL_OPT_COMPRESS, NULLPTR) != 0){
			THROW_MYSQL_EXCEPTION;
		}
		const ::my_bool TRUE_VALUE = true;
		if(::mysql_options(m_mysql.get(), MYSQL_OPT_RECONNECT, &TRUE_VALUE) != 0){
			THROW_MYSQL_EXCEPTION;
		}
		if(::mysql_options(m_mysql.get(), MYSQL_SET_CHARSET_NAME, charset.c_str()) != 0){
			THROW_MYSQL_EXCEPTION;
		}

		if(!::mysql_real_connect(m_mysql.get(), serverAddr.c_str(), userName.c_str(),
			password.c_str(), schema.c_str(), serverPort, NULLPTR, useSsl ? CLIENT_SSL : 0))
		{
			THROW_MYSQL_EXCEPTION;
		}
	}

public:
	void executeSql(const std::string &sql){
		m_result.reset();
		m_columns.clear();
		m_row = NULLPTR;

		if(::mysql_real_query(m_mysql.get(), sql.data(), sql.size()) != 0){
			THROW_MYSQL_EXCEPTION;
		}

		if(!m_result.reset(::mysql_use_result(m_mysql.get()))){
			if(::mysql_errno(m_mysql.get()) != 0){
				THROW_MYSQL_EXCEPTION;
			}
			// 没有返回结果。
		} else {
			const AUTO(fields, ::mysql_fetch_fields(m_result.get()));
			const AUTO(count, ::mysql_num_fields(m_result.get()));
			m_columns.reserve(count);
			m_columns.clear();
			for(std::size_t i = 0; i < count; ++i){
				MySqlColumns::iterator ins;
				const char *const name = fields[i].name;
				if(lowerBoundColumn(ins, m_columns, name)){
					LOG_POSEIDON_ERROR("Duplicate column in MySQL result set: ", name);
					DEBUG_THROW(Exception, SharedNts::observe("Duplicate column"));
				}
				LOG_POSEIDON_TRACE("MySQL result column: name = ", name, ", index = ", i);
				m_columns.insert(ins, std::make_pair(name, i));
			}
		}
	}

	boost::uint64_t getInsertId() const {
		return ::mysql_insert_id(m_mysql.get());
	}

	bool fetchRow(){
		if(m_columns.empty()){
			LOG_POSEIDON_DEBUG("Empty set returned form MySQL server.");
			return false;
		}
		m_row = ::mysql_fetch_row(m_result.get());
		if(!m_row){
			if(::mysql_errno(m_mysql.get()) != 0){
				THROW_MYSQL_EXCEPTION;
			}
			return false;
		}
		m_lengths = ::mysql_fetch_lengths(m_result.get());
		return true;
	}
	boost::int64_t getSigned(const char *column) const {
		MySqlColumns::const_iterator it;
		if(!lowerBoundColumn(it, m_columns, column)){
			LOG_POSEIDON_ERROR("Column not found: ", column);
			DEBUG_THROW(Exception, SharedNts::observe("Column not found"));
		}
		const AUTO(data, m_row[it->second]);
		if(!data || (data[0] == 0)){
			return 0;
		}
		char *endptr;
		const AUTO(val, ::strtoll(data, &endptr, 10));
		if(endptr[0] != 0){
			LOG_POSEIDON_ERROR("Could not convert column data to long long: ", data);
			DEBUG_THROW(Exception, SharedNts::observe("Invalid data format"));
		}
		return val;
	}
	boost::uint64_t getUnsigned(const char *column) const {
		MySqlColumns::const_iterator it;
		if(!lowerBoundColumn(it, m_columns, column)){
			LOG_POSEIDON_ERROR("Column not found: ", column);
			DEBUG_THROW(Exception, SharedNts::observe("Column not found"));
		}
		const AUTO(data, m_row[it->second]);
		if(!data || (data[0] == 0)){
			return 0;
		}
		char *endptr;
		const AUTO(val, ::strtoull(data, &endptr, 10));
		if(endptr[0] != 0){
			LOG_POSEIDON_ERROR("Could not convert column data to unsigned long long: ", data);
			DEBUG_THROW(Exception, SharedNts::observe("Invalid data format"));
		}
		return val;
	}
	double getDouble(const char *column) const {
		MySqlColumns::const_iterator it;
		if(!lowerBoundColumn(it, m_columns, column)){
			LOG_POSEIDON_ERROR("Column not found: ", column);
			DEBUG_THROW(Exception, SharedNts::observe("Column not found"));
		}
		const AUTO(data, m_row[it->second]);
		if(!data || (data[0] == 0)){
			return 0;
		}
		char *endptr;
		const AUTO(val, ::strtod(data, &endptr));
		if(endptr[0] != 0){
			LOG_POSEIDON_ERROR("Could not convert column data to double: ", data);
			DEBUG_THROW(Exception, SharedNts::observe("Invalid data format"));
		}
		return val;
	}
	std::string getString(const char *column) const {
		MySqlColumns::const_iterator it;
		if(!lowerBoundColumn(it, m_columns, column)){
			LOG_POSEIDON_ERROR("Column not found: ", column);
			DEBUG_THROW(Exception, SharedNts::observe("Column not found"));
		}
		std::string val;
		const AUTO(data, m_row[it->second]);
		if(data){
			val.assign(data, m_lengths[it->second]);
		}
		return val;
	}
};

}

void MySqlConnection::create(boost::scoped_ptr<MySqlConnection> &conn,
	MySqlThreadContext & /* context */, const std::string &serverAddr, unsigned serverPort,
	const std::string &userName, const std::string &password, const std::string &schema,
	bool useSsl, const std::string &charset)
{
	conn.reset(new MySqlConnectionDelegate(
		serverAddr, serverPort, userName, password, schema, useSsl, charset));
}

MySqlConnection::~MySqlConnection(){
}

void MySqlConnection::executeSql(const std::string &sql){
	static_cast<MySqlConnectionDelegate *>(this)->executeSql(sql);
}

boost::uint64_t MySqlConnection::getInsertId() const {
	return static_cast<const MySqlConnectionDelegate *>(this)->getInsertId();
}

bool MySqlConnection::fetchRow(){
	return static_cast<MySqlConnectionDelegate *>(this)->fetchRow();
}
boost::int64_t MySqlConnection::getSigned(const char *column) const {
	return static_cast<const MySqlConnectionDelegate *>(this)->getSigned(column);
}
boost::uint64_t MySqlConnection::getUnsigned(const char *column) const {
	return static_cast<const MySqlConnectionDelegate *>(this)->getUnsigned(column);
}
double MySqlConnection::getDouble(const char *column) const {
	return static_cast<const MySqlConnectionDelegate *>(this)->getDouble(column);
}
std::string MySqlConnection::getString(const char *column) const {
	return static_cast<const MySqlConnectionDelegate *>(this)->getString(column);
}
