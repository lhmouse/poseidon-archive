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

	bool m_connected_notified;
	bool m_read_hup_notified;

	mutable Mutex m_send_mutex;
	StreamBuffer m_send_buffer;

	volatile boost::uint64_t m_shutdown_time;
	volatile boost::uint64_t m_last_use_time;
	mutable Mutex m_shutdown_mutex;
	boost::shared_ptr<TimerItem> m_shutdown_timer;

public:
	explicit TcpSessionBase(UniqueFile socket);
	~TcpSessionBase();

protected:
	void init_ssl(Move<boost::scoped_ptr<SslFilterBase> > ssl_filter);
	void create_shutdown_timer();

	// 注意，只能在 epoll 线程中调用这些函数。
	int poll_read_and_process(bool readable) OVERRIDE;
	int poll_write(Mutex::UniqueLock &write_lock, bool writeable) OVERRIDE;

	void on_connect() OVERRIDE = 0;
	void on_read_hup() OVERRIDE = 0;
	void on_close(int err_code) OVERRIDE = 0; // 参数就是 errno。
	void on_receive(StreamBuffer data) OVERRIDE = 0;

public:
	bool has_been_shutdown_read() const NOEXCEPT OVERRIDE;
	bool has_been_shutdown_write() const NOEXCEPT OVERRIDE;
	bool shutdown_read() NOEXCEPT OVERRIDE;
	bool shutdown_write() NOEXCEPT OVERRIDE;
	void force_shutdown() NOEXCEPT OVERRIDE;

	bool is_throttled() const OVERRIDE;

	bool is_using_ssl() const;

	void set_no_delay(bool enabled = true);
	void set_timeout(boost::uint64_t timeout);

	bool send(StreamBuffer buffer) OVERRIDE;
};

}

#endif
