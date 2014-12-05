// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_CONNECTION_HPP_
#define POSEIDON_MYSQL_CONNECTION_HPP_

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/cstdint.hpp>
#include <boost/scoped_ptr.hpp>

namespace Poseidon {

class MySqlThreadContext;

class MySqlConnection : boost::noncopyable {
public:
	static void create(boost::scoped_ptr<MySqlConnection> &conn,
		MySqlThreadContext &context, const std::string &serverAddr, unsigned serverPort,
		const std::string &userName, const std::string &password, const std::string &schema,
		bool useSsl, const std::string &charset);

public:
	virtual ~MySqlConnection() = 0;

public:
	void executeSql(const std::string &sql);

	boost::uint64_t getInsertId() const;

	bool fetchRow();
	boost::int64_t getSigned(const char *column) const;
	boost::uint64_t getUnsigned(const char *column) const;
	double getDouble(const char *column) const;
	std::string getString(const char *column) const;
};

}

#endif
