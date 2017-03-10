// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UDP_SERVER_BASE_HPP_
#define POSEIDON_UDP_SERVER_BASE_HPP_

#include <utility>
#include <boost/scoped_ptr.hpp>
#include <boost/container/deque.hpp>
#include "socket_server_base.hpp"
#include "sock_addr.hpp"
#include "mutex.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

class UdpServerBase : public SocketServerBase {
private:
	struct SyncIoResult {
		long bytes_transferred;
		int err_code;
	};

private:
	volatile bool m_shutdown_read;
	volatile bool m_shutdown_write;

	mutable Mutex m_send_mutex;
	mutable boost::container::deque<std::pair<SockAddr, StreamBuffer> > m_send_queue;

public:
	explicit UdpServerBase(const SockAddr &addr);
	explicit UdpServerBase(const IpPort &addr);
	~UdpServerBase();

private:
	SyncIoResult sync_read_and_process(void *hint, unsigned long hint_size) const;
	SyncIoResult sync_write(void *hint, unsigned long hint_size) const;

protected:
	// 注意，只能在 epoll 线程中调用这些函数。
	virtual void on_receive(const SockAddr &sock_addr, StreamBuffer data) const = 0;

public:
	bool has_been_shutdown_read() const NOEXCEPT;
	bool has_been_shutdown_write() const NOEXCEPT;
	bool shutdown_read() NOEXCEPT;
	bool shutdown_write() NOEXCEPT;
	void force_shutdown() NOEXCEPT;

	bool poll() const OVERRIDE;

	bool send(const SockAddr &sock_addr, StreamBuffer buffer) const;
};

}

#endif
