// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UDP_SERVER_BASE_HPP_
#define POSEIDON_UDP_SERVER_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/container/deque.hpp>
#include "socket_base.hpp"
#include "sock_addr.hpp"
#include "ip_port.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

class UdpServerBase : public SocketBase {

private:
	mutable Mutex m_send_mutex;
	mutable boost::container::deque<std::pair<SockAddr, StreamBuffer> > m_send_queue;

public:
	explicit UdpServerBase(const SockAddr &addr);
	~UdpServerBase();

protected:
	// 注意，只能在 epoll 线程中调用这些函数。
	int poll_read_and_process(bool readable) OVERRIDE;
	int poll_write(Mutex::UniqueLock &write_lock, bool writeable) OVERRIDE;

	virtual void on_receive(const SockAddr &sock_addr, StreamBuffer data) const = 0;
	virtual void on_message_too_large(const SockAddr &sock_addr, StreamBuffer data) const;

public:
	bool has_been_shutdown_read() const NOEXCEPT {
		return SocketBase::has_been_shutdown_read();
	}
	bool has_been_shutdown_write() const NOEXCEPT {
		return SocketBase::has_been_shutdown_write();
	}
	bool shutdown_read() NOEXCEPT {
		return SocketBase::shutdown_read();
	}
	bool shutdown_write() NOEXCEPT {
		return SocketBase::shutdown_write();
	}
	void force_shutdown() NOEXCEPT {
		return SocketBase::force_shutdown();
	}

	bool send(const SockAddr &sock_addr, StreamBuffer buffer) const;
};

}

#endif
