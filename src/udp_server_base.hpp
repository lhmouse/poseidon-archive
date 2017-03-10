// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UDP_SERVER_BASE_HPP_
#define POSEIDON_UDP_SERVER_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include "socket_server_base.hpp"
#include "sock_addr.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

class UdpServerBase : public SocketServerBase {
public:
	explicit UdpServerBase(const SockAddr &addr);
	explicit UdpServerBase(const IpPort &addr);
	~UdpServerBase();

protected:
	// 注意，只能在 epoll 线程中调用这些函数。
	virtual void on_receive(const SockAddr &sock_addr, StreamBuffer data) const = 0;

public:
	bool poll() const OVERRIDE;

	bool send(const SockAddr &sock_addr, StreamBuffer data) const;
};

}

#endif
