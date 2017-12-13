// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_CLIENT_BASE_HPP_
#define POSEIDON_TCP_CLIENT_BASE_HPP_

#include "tcp_session_base.hpp"

namespace Poseidon {

class ClientSslFactory;

class TcpClientBase : public TcpSessionBase {
private:
	boost::scoped_ptr<ClientSslFactory> m_ssl_factory;

public:
	explicit TcpClientBase(const SockAddr &addr, bool use_ssl = false, bool verify_peer = true);
	~TcpClientBase();
};

}

#endif
