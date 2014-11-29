// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "connection.hpp"
#define POSEIDON_MYSQL_CONNECTION_IMPL_
#include "connection_impl.hpp"
#include "exception.hpp"
#include <string.h>
#include <stdlib.h>
#include "../log.hpp"
using namespace Poseidon;

namespace {

// 若返回 true 则 pos 指向找到的列，否则 pos 指向下一个。
template<typename IteratorT, typename VectorT>
bool findColumn(IteratorT &pos, VectorT &columns, const char *name){
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

}

#define THROW_MYSQL_EXCEPTION	\
	DEBUG_THROW(MySqlException, ::mysql_errno(this->m_mySql.get()), ::mysql_error(this->m_mySql.get()))

MySqlConnectionImpl::MySqlConnectionImpl(const std::string &serverAddr, unsigned serverPort,
	const std::string &userName, const std::string &password, const std::string &schema,
	bool useSsl, const std::string &charset)
	: m_row(NULLPTR), m_lengths(NULLPTR)
{
	if(!m_mySql.reset(::mysql_init(&m_mySqlObject))){
		DEBUG_THROW(SystemError, ENOMEM);
	}

	if(::mysql_options(m_mySql.get(), MYSQL_OPT_COMPRESS, NULLPTR) != 0){
		THROW_MYSQL_EXCEPTION;
	}
	const ::my_bool trueVal = true;
	if(::mysql_options(m_mySql.get(), MYSQL_OPT_RECONNECT, &trueVal) != 0){
		THROW_MYSQL_EXCEPTION;
	}
	if(::mysql_options(m_mySql.get(), MYSQL_SET_CHARSET_NAME, charset.c_str()) != 0){
		THROW_MYSQL_EXCEPTION;
	}

	if(!::mysql_real_connect(m_mySql.get(), serverAddr.c_str(), userName.c_str(),
		password.c_str(), schema.c_str(), serverPort, NULLPTR, useSsl ? CLIENT_SSL : 0))
	{
		THROW_MYSQL_EXCEPTION;
	}
}
MySqlConnectionImpl::~MySqlConnectionImpl(){
}

void MySqlConnectionImpl::executeSql(const std::string &sql){
	m_result.reset();
	m_columns.clear();
	m_row = NULLPTR;

	if(::mysql_real_query(m_mySql.get(), sql.data(), sql.size()) != 0){
		THROW_MYSQL_EXCEPTION;
	}
}
void MySqlConnectionImpl::waitForResult(){
	if(!m_result.reset(::mysql_use_result(m_mySql.get()))){
		THROW_MYSQL_EXCEPTION;
	}

	const AUTO(fields, ::mysql_fetch_fields(m_result.get()));
	const AUTO(count, ::mysql_num_fields(m_result.get()));
	m_columns.reserve(count);
	m_columns.clear();
	for(std::size_t i = 0; i < count; ++i){
		MySqlColumns::iterator ins;
		const char *const name = fields[i].name;
		if(findColumn(ins, m_columns, name)){
			LOG_POSEIDON_ERROR("Duplicate column in MySQL result set: ", name);
			DEBUG_THROW(Exception, "Duplicate column");
		}
		LOG_POSEIDON_TRACE("MySQL result column: name = ", name, ", index = ", i);
		m_columns.insert(ins, std::make_pair(name, i));
	}
}

bool MySqlConnectionImpl::fetchRow(){
	if(m_columns.empty()){
		LOG_POSEIDON_DEBUG("Empty set returned form MySQL server.");
		return false;
	}
	m_row = ::mysql_fetch_row(m_result.get());
	if(!m_row){
		if(::mysql_errno(m_mySql.get()) != 0){
			THROW_MYSQL_EXCEPTION;
		}
		return false;
	}
	m_lengths = ::mysql_fetch_lengths(m_result.get());
	return true;
}
boost::int64_t MySqlConnectionImpl::getSigned(const char *column) const {
	MySqlColumns::const_iterator it;
	if(!findColumn(it, m_columns, column)){
		LOG_POSEIDON_ERROR("Column not found: ", column);
		DEBUG_THROW(Exception, "Column not found");
	}
	const AUTO(data, m_row[it->second]);
	if(!data || (data[0] == 0)){
		return 0;
	}
	char *endptr;
	const AUTO(val, ::strtoll(data, &endptr, 10));
	if(endptr[0] != 0){
		LOG_POSEIDON_ERROR("Could not convert column data to long long: ", data);
		DEBUG_THROW(Exception, "Invalid data format");
	}
	return val;
}
boost::uint64_t MySqlConnectionImpl::getUnsigned(const char *column) const {
	MySqlColumns::const_iterator it;
	if(!findColumn(it, m_columns, column)){
		LOG_POSEIDON_ERROR("Column not found: ", column);
		DEBUG_THROW(Exception, "Column not found");
	}
	const AUTO(data, m_row[it->second]);
	if(!data || (data[0] == 0)){
		return 0;
	}
	char *endptr;
	const AUTO(val, ::strtoull(data, &endptr, 10));
	if(endptr[0] != 0){
		LOG_POSEIDON_ERROR("Could not convert column data to unsigned long long: ", data);
		DEBUG_THROW(Exception, "Invalid data format");
	}
	return val;
}
double MySqlConnectionImpl::getDouble(const char *column) const {
	MySqlColumns::const_iterator it;
	if(!findColumn(it, m_columns, column)){
		LOG_POSEIDON_ERROR("Column not found: ", column);
		DEBUG_THROW(Exception, "Column not found");
	}
	const AUTO(data, m_row[it->second]);
	if(!data || (data[0] == 0)){
		return 0;
	}
	char *endptr;
	const AUTO(val, ::strtod(data, &endptr));
	if(endptr[0] != 0){
		LOG_POSEIDON_ERROR("Could not convert column data to double: ", data);
		DEBUG_THROW(Exception, "Invalid data format");
	}
	return val;
}
std::string MySqlConnectionImpl::getString(const char *column) const {
	MySqlColumns::const_iterator it;
	if(!findColumn(it, m_columns, column)){
		LOG_POSEIDON_ERROR("Column not found: ", column);
		DEBUG_THROW(Exception, "Column not found");
	}
	std::string val;
	const AUTO(data, m_row[it->second]);
	if(data){
		val.assign(data, m_lengths[it->second]);
	}
	return val;
}
