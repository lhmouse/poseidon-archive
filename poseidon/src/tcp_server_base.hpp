// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_SERVER_BASE_HPP_
#define POSEIDON_TCP_SERVER_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include "socket_base.hpp"
#include "sock_addr.hpp"
#include "ip_port.hpp"

namespace Poseidon {

class Ssl_server_factory;
class Tcp_session_base;

// 抽象工厂模式
class Tcp_server_base : public Socket_base {
private:
	boost::scoped_ptr<Ssl_server_factory> m_ssl_factory;

public:
	explicit Tcp_server_base(const Sock_addr &addr, const char *certificate = "", const char *private_key = "");
	~Tcp_server_base();

protected:
	// 工厂函数。返回空指针导致抛出一个异常。
	virtual boost::shared_ptr<Tcp_session_base> on_client_connect(Move<Unique_file> client) = 0;

public:
	int poll_read_and_process(unsigned char *hint_buffer, std::size_t hint_capacity, bool readable) OVERRIDE;
};

}

#endif
