// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_CLIENT_BASE_HPP_
#define POSEIDON_TCP_CLIENT_BASE_HPP_

#include "tcp_session_base.hpp"
#include <string>
#include <boost/shared_ptr.hpp>
#include "sock_addr.hpp"
#include "shared_ntmbs.hpp"

namespace Poseidon {

class IpPort;

class TcpClientBase : public TcpSessionBase {
private:
	class SslImplClient;

private:
	SockAddr m_sockAddr;

protected:
	explicit TcpClientBase(const IpPort &addr);

protected:
	void onReadAvail(const void *data, std::size_t size) = 0;

protected:
	void sslConnect();
	void goResident();
};

}

#endif
