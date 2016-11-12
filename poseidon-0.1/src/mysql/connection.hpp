// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_CONNECTION_HPP_
#define POSEIDON_MYSQL_CONNECTION_HPP_

#include "../cxx_util.hpp"
#include <string>
#include <cstring>
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class Uuid;

namespace MySql {
	class Connection : NONCOPYABLE {
	public:
		static boost::shared_ptr<Connection> create(const char *server_addr, unsigned server_port,
			const char *user_name, const char *password, const char *schema, bool use_ssl, const char *charset);

		static boost::shared_ptr<Connection> create(const std::string &server_addr, unsigned server_port,
			const std::string &user_name, const std::string &password, const std::string &schema, bool use_ssl, const std::string &charset)
		{
			return create(server_addr.c_str(), server_port,
				user_name.c_str(), password.c_str(), schema.c_str(), use_ssl, charset.c_str());
		}

	public:
		virtual ~Connection() = 0;

	public:
		void execute_sql(const char *sql, std::size_t len);
		void execute_sql(const char *sql){
			execute_sql(sql, std::strlen(sql));
		}
		void execute_sql(const std::string &sql){
			execute_sql(sql.data(), sql.size());
		}
		void discard_result() NOEXCEPT;

		boost::uint64_t get_insert_id() const;
		bool fetch_row();

		boost::int64_t get_signed(const char *column) const;
		boost::uint64_t get_unsigned(const char *column) const;
		double get_double(const char *column) const;
		std::string get_string(const char *column) const;
		boost::uint64_t get_datetime(const char *column) const;
		Uuid get_uuid(const char *column) const;
	};
}

}

#endif
