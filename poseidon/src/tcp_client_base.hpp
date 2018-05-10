// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_CLIENT_BASE_HPP_
#define POSEIDON_TCP_CLIENT_BASE_HPP_

#include "tcp_session_base.hpp"

namespace Poseidon {

class Ssl_client_factory;

class Tcp_client_base : public Tcp_session_base {
private:
	boost::scoped_ptr<Ssl_client_factory> m_ssl_factory;

public:
	explicit Tcp_client_base(const Sock_addr &addr, bool use_ssl = false, bool verify_peer = true);
	~Tcp_client_base();
};

}

#endif
