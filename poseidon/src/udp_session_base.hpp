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
#include <mutex>

namespace Poseidon {

class Udp_session_base : public Socket_base {
private:
	mutable std::mutex m_send_mutex;
	mutable boost::container::deque<std::pair<Sock_addr, Stream_buffer> > m_send_queue;

public:
	explicit Udp_session_base(Move<Unique_file> socket);
	~Udp_session_base();

protected:
	// 注意，只能在 epoll 线程中调用这些函数。
	int poll_read_and_process(unsigned char *hint_buffer, std::size_t hint_capacity, bool readable) OVERRIDE;
	int poll_write(std::unique_lock<std::mutex> &write_lock, unsigned char *hint_buffer, std::size_t hint_capacity, bool writable) OVERRIDE;

	virtual void on_receive(const Sock_addr &sock_addr, Stream_buffer data) = 0;
	virtual void on_message_too_large(const Sock_addr &sock_addr, Stream_buffer data);

public:
	bool has_been_shutdown_read() const NOEXCEPT OVERRIDE;
	bool has_been_shutdown_write() const NOEXCEPT OVERRIDE;
	bool shutdown_read() NOEXCEPT OVERRIDE;
	bool shutdown_write() NOEXCEPT OVERRIDE;
	void force_shutdown() NOEXCEPT OVERRIDE;

	void add_membership(const Sock_addr &group);
	void drop_membership(const Sock_addr &group);
	void set_multicast_loop(bool enabled = true);
	void set_multicast_ttl(int ttl);

	bool send(const Sock_addr &sock_addr, Stream_buffer buffer);
};

}

#endif
