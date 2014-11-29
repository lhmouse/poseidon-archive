// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_CONNECTION_HPP_
#define POSEIDON_MYSQL_CONNECTION_HPP_

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class MySqlConnection : boost::noncopyable {
protected:
	virtual ~MySqlConnection() = 0;

public:
	virtual void executeSql(const std::string &sql) = 0;
	virtual void waitForResult() = 0;

	virtual bool fetchRow() = 0;
	virtual boost::int64_t getSigned(const char *column) const = 0;
	virtual boost::uint64_t getUnsigned(const char *column) const = 0;
	virtual double getDouble(const char *column) const = 0;
	virtual std::string getString(const char *column) const = 0;
};

}

#endif
