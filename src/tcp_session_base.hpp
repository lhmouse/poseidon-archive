// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_SESSION_BASE_HPP_
#define POSEIDON_TCP_SESSION_BASE_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "socket_base.hpp"
#include "session_base.hpp"
#include <boost/scoped_ptr.hpp>

namespace Poseidon {

class TcpServerBase;
class SslFilterBase;
class TimerItem;

class TcpSessionBase : public SocketBase, public SessionBase {
	friend TcpServerBase;

private:
	static void shutdown_timer_proc(const boost::weak_ptr<TcpSessionBase> &weak, boost::uint64_t now);

private:
	boost::scoped_ptr<SslFilterBase> m_ssl_filter;

	volatile bool m_connected;
	bool m_connected_notified;
	bool m_read_hup_notified;

	mutable Mutex m_send_mutex;
	StreamBuffer m_send_buffer;

	volatile boost::uint64_t m_shutdown_time;
	mutable Mutex m_shutdown_mutex;
	boost::shared_ptr<TimerItem> m_shutdown_timer;

public:
	explicit TcpSessionBase(UniqueFile socket);
	~TcpSessionBase();

protected:
	void init_ssl(Move<boost::scoped_ptr<SslFilterBase> > ssl_filter);

	// 注意，只能在 epoll 线程中调用这些函数。
	int poll_read_and_process() OVERRIDE;
	int poll_write(Mutex::UniqueLock &socket_lock) OVERRIDE;

	virtual void on_connect();
	void on_read_hup() NOEXCEPT OVERRIDE;
	void on_close(int err_code) NOEXCEPT OVERRIDE; // 参数就是 errno。

	void on_receive(StreamBuffer data) OVERRIDE = 0;

public:
	bool has_been_shutdown_read() const NOEXCEPT OVERRIDE {
		return SocketBase::has_been_shutdown_read();
	}
	bool has_been_shutdown_write() const NOEXCEPT OVERRIDE {
		return SocketBase::has_been_shutdown_write();
	}
	bool shutdown_read() NOEXCEPT OVERRIDE {
		return SocketBase::shutdown_read();
	}
	bool shutdown_write() NOEXCEPT OVERRIDE {
		return SocketBase::shutdown_write();
	}
	void force_shutdown() NOEXCEPT OVERRIDE {
		return SocketBase::force_shutdown();
	}

	bool is_using_ssl() const {
		return !!m_ssl_filter;
	}

	bool is_throttled() const OVERRIDE;
	bool is_connected() const NOEXCEPT;

	void set_no_delay(bool enabled = true);
	void set_timeout(boost::uint64_t timeout);

	bool send(StreamBuffer buffer);
};

}

#endif
