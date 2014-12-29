// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_CONNECTION_HPP_
#define POSEIDON_MYSQL_CONNECTION_HPP_

#include "../cxx_util.hpp"
#include <string>
#include <boost/cstdint.hpp>
#include <boost/scoped_ptr.hpp>

namespace Poseidon {

class MySqlThreadContext;

class MySqlConnection : NONCOPYABLE {
public:
	static void create(boost::scoped_ptr<MySqlConnection> &conn,
		MySqlThreadContext &context, const char *serverAddr, unsigned serverPort,
		const char *userName, const char *password, const char *schema,
		bool useSsl, const char *charset);

	static void create(boost::scoped_ptr<MySqlConnection> &conn,
		MySqlThreadContext &context, const std::string &serverAddr, unsigned serverPort,
		const std::string &userName, const std::string &password, const std::string &schema,
		bool useSsl, const std::string &charset)
	{
		create(conn, context, serverAddr.c_str(), serverPort,
			userName.c_str(), password.c_str(), schema.c_str(), useSsl, charset.c_str());
	}

public:
	virtual ~MySqlConnection() = 0;

public:
	virtual void executeSql(const std::string &sql) = 0;

	virtual boost::uint64_t getInsertId() const = 0;

	virtual bool fetchRow() = 0;
	virtual boost::int64_t getSigned(const char *column) const = 0;
	virtual boost::uint64_t getUnsigned(const char *column) const = 0;
	virtual double getDouble(const char *column) const = 0;
	virtual std::string getString(const char *column) const = 0;
	virtual double getDateTime(const char *column) const = 0;
};

}

#endif
