// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UDP_SESSION_BASE_HPP_
#define POSEIDON_UDP_SESSION_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/container/deque.hpp>
#include "socket_base.hpp"
#include "sock_addr.hpp"
#include "ip_port.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

class UdpSessionBase : public SocketBase {
private:
	mutable Mutex m_send_mutex;
	mutable boost::container::deque<std::pair<SockAddr, StreamBuffer> > m_send_queue;

public:
	explicit UdpSessionBase(Move<UniqueFile> socket);
	~UdpSessionBase();

protected:
	// 注意，只能在 epoll 线程中调用这些函数。
	int poll_read_and_process(unsigned char *hint_buffer, std::size_t hint_capacity, bool readable) OVERRIDE;
	int poll_write(Mutex::UniqueLock &write_lock, unsigned char *hint_buffer, std::size_t hint_capacity, bool writeable) OVERRIDE;

	virtual void on_receive(const SockAddr &sock_addr, StreamBuffer data) = 0;
	virtual void on_message_too_large(const SockAddr &sock_addr, StreamBuffer data);

public:
	bool has_been_shutdown_read() const NOEXCEPT OVERRIDE;
	bool has_been_shutdown_write() const NOEXCEPT OVERRIDE;
	bool shutdown_read() NOEXCEPT OVERRIDE;
	bool shutdown_write() NOEXCEPT OVERRIDE;
	void force_shutdown() NOEXCEPT OVERRIDE;

	bool send(const SockAddr &sock_addr, StreamBuffer buffer);
};

}

#endif
