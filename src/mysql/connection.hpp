// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_CONNECTION_HPP_
#define POSEIDON_MYSQL_CONNECTION_HPP_

#include "../cxx_util.hpp"
#include <string>
#include <cstring>
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

namespace MySql {
	class Connection : NONCOPYABLE {
	public:
		static boost::shared_ptr<Connection> create(const char *serverAddr, unsigned serverPort,
			const char *userName, const char *password, const char *schema,
			bool useSsl, const char *charset);

		static boost::shared_ptr<Connection> create(const std::string &serverAddr, unsigned serverPort,
			const std::string &userName, const std::string &password, const std::string &schema,
			bool useSsl, const std::string &charset)
		{
			return create(serverAddr.c_str(), serverPort,
				userName.c_str(), password.c_str(), schema.c_str(), useSsl, charset.c_str());
		}

	public:
		virtual ~Connection() = 0;

	public:
		void executeSql(const char *sql, std::size_t len);
		void executeSql(const char *sql){
			executeSql(sql, std::strlen(sql));
		}
		void executeSql(const std::string &sql){
			executeSql(sql.data(), sql.size());
		}

		boost::uint64_t getInsertId() const;
		bool fetchRow();

		boost::int64_t getSigned(const char *column) const;
		boost::uint64_t getUnsigned(const char *column) const;
		double getDouble(const char *column) const;
		std::string getString(const char *column) const;
		boost::uint64_t getDateTime(const char *column) const;
	};
}

}

#endif
