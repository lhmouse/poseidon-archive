// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_SESSION_BASE_HPP_
#define POSEIDON_TCP_SESSION_BASE_HPP_

#include "cxx_ver.hpp"
#include "socket_base.hpp"
#include "session_base.hpp"
#include <mutex>
#include <boost/scoped_ptr.hpp>

namespace Poseidon {

class Tcp_server_base;
class Tcp_client_base;
class Ssl_filter;
class Timer;

class Tcp_session_base : public Socket_base, public Session_base {
	friend Tcp_server_base;
	friend Tcp_client_base;

private:
	static void shutdown_timer_proc(const boost::weak_ptr<Tcp_session_base> &weak, std::uint64_t now);

private:
	boost::scoped_ptr<Ssl_filter> m_ssl_filter;

	bool m_connected_notified;
	bool m_read_hup_notified;

	mutable std::mutex m_send_mutex;
	Stream_buffer m_send_buffer;

	volatile std::uint64_t m_shutdown_time;
	volatile std::uint64_t m_last_use_time;
	mutable std::mutex m_shutdown_mutex;
	boost::shared_ptr<Timer> m_shutdown_timer;

public:
	explicit Tcp_session_base(Move<Unique_file> socket);
	~Tcp_session_base();

private:
	void init_ssl(boost::scoped_ptr<Ssl_filter> &ssl_filter);
	void create_shutdown_timer();

protected:
	// 注意，只能在 epoll 线程中调用这些函数。
	int poll_read_and_process(unsigned char *hint_buffer, std::size_t hint_capacity, bool readable) OVERRIDE;
	int poll_write(std::unique_lock<std::mutex> &write_lock, unsigned char *hint_buffer, std::size_t hint_capacity, bool writable) OVERRIDE;

	void on_connect() OVERRIDE = 0;
	void on_read_hup() OVERRIDE = 0;
	void on_close(int err_code) OVERRIDE = 0; // 参数就是 errno。
	void on_receive(Stream_buffer data) OVERRIDE = 0;

	// 注意，只能在 timer 线程中调用这些函数。
	virtual void on_shutdown_timer(std::uint64_t now);

public:
	bool has_been_shutdown_read() const NOEXCEPT OVERRIDE;
	bool has_been_shutdown_write() const NOEXCEPT OVERRIDE;
	bool shutdown_read() NOEXCEPT OVERRIDE;
	bool shutdown_write() NOEXCEPT OVERRIDE;
	void force_shutdown() NOEXCEPT OVERRIDE;

	bool is_using_ssl() const;
	bool is_throttled() const OVERRIDE;

	void set_no_delay(bool enabled = true);
	void set_timeout(std::uint64_t timeout);

	bool send(Stream_buffer buffer) OVERRIDE;
};

}

#endif
