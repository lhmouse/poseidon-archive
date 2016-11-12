// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_SESSION_BASE_HPP_
#define POSEIDON_TCP_SESSION_BASE_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "session_base.hpp"
#include <string>
#include <deque>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/cstdint.hpp>
#include "mutex.hpp"
#include "raii.hpp"
#include "ip_port.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

class Epoll;
class SslFilterBase;
class TimerItem;

class TcpServerBase;
class TcpClientBase;

class TcpSessionBase : public SessionBase {
	friend Epoll;

	friend TcpServerBase;
	friend TcpClientBase;

private:
	struct SyncIoResult {
		long bytes_transferred;
		int err_code;
	};

public:
	// 至少一个此对象存活的条件下连接不会由于 RDHUP 而被关掉。
	class DelayedShutdownGuard : NONCOPYABLE {
	private:
		const boost::weak_ptr<TcpSessionBase> m_weak;

	public:
		explicit DelayedShutdownGuard(boost::weak_ptr<TcpSessionBase> weak);
		~DelayedShutdownGuard();
	};

private:
	static void shutdown_timer_proc(const boost::weak_ptr<TcpSessionBase> &weak, boost::uint64_t now);

private:
	const UniqueFile m_socket;
	const boost::uint64_t m_created_time;

	mutable struct {
		Mutex mutex;
		IpPort remote;
		IpPort local;
	} m_peer_info;

	volatile bool m_connected;
	volatile bool m_connected_notified;
	boost::scoped_ptr<SslFilterBase> m_ssl_filter;

	volatile bool m_shutdown_read;
	volatile bool m_read_hup_notified;
	volatile bool m_shutdown_write;
	volatile bool m_really_shutdown_write;
	volatile bool m_timed_out;
	volatile bool m_throttled;
	volatile std::size_t m_delayed_shutdown_guard_count;

	mutable Mutex m_buffer_mutex;
	StreamBuffer m_send_buffer;
	boost::weak_ptr<Epoll> m_epoll;

	volatile boost::uint64_t m_shutdown_time;
	mutable Mutex m_timer_mutex;
	boost::shared_ptr<TimerItem> m_shutdown_timer;

protected:
	explicit TcpSessionBase(UniqueFile socket);
	~TcpSessionBase();

private:
	bool has_connected() const NOEXCEPT;
	void set_connected();
	bool is_connected_notified() const NOEXCEPT;
	void notify_connected() NOEXCEPT;

	void init_ssl(Move<boost::scoped_ptr<SslFilterBase> > ssl_filter);

	bool is_read_hup_notified() const NOEXCEPT;
	void notify_read_hup() NOEXCEPT;

	void set_epoll(boost::weak_ptr<Epoll> epoll) NOEXCEPT;
	void notify_epoll_writeable() NOEXCEPT;

	// 同步，线程安全。
	void fetch_peer_info() const;
	// 和 Windows 的 IsDialogMessage() 类似，这个函数读取并在内部调用 on_read_avail() 处理数据。
	// 这里的出参返回读取的数据，一次性读取的字节数不大于 hint_size。如果开启了 SSL，返回明文。
	SyncIoResult sync_read_and_process(void *hint, unsigned long hint_size);
	// 这里的出参返回写入的数据，一次性写入的字节数不大于 hint_size。如果开启了 SSL，返回明文。
	SyncIoResult sync_write(void *hint, unsigned long hint_size);
	// 出参用于确保 epoll 和写入内部缓冲的顺序。
	std::size_t get_send_buffer_size(Mutex::UniqueLock *lock = NULLPTR) const;

protected:
	virtual void on_connect();

	void on_read_hup() NOEXCEPT OVERRIDE;
	void on_close(int err_code) NOEXCEPT OVERRIDE; // 参数就是 errno。

	// 注意，只能在 epoll 线程中调用这些函数。
	void on_read_avail(StreamBuffer data) OVERRIDE = 0;

	bool send(StreamBuffer buffer) OVERRIDE;

public:
	bool has_been_shutdown_read() const NOEXCEPT OVERRIDE;
	bool has_been_shutdown_write() const NOEXCEPT OVERRIDE;
	bool shutdown_read() NOEXCEPT OVERRIDE;
	bool shutdown_write() NOEXCEPT OVERRIDE;
	void force_shutdown() NOEXCEPT OVERRIDE;

	int get_fd() const {
		return m_socket.get();
	}

	bool is_using_ssl() const {
		return !!m_ssl_filter;
	}

	boost::uint64_t get_created_time() const {
		return m_created_time;
	}

	const IpPort &get_remote_info() const;
	const IpPort &get_local_info() const;
	IpPort get_remote_info_nothrow() const NOEXCEPT;
	IpPort get_local_info_nothrow() const NOEXCEPT;

	void set_timeout(boost::uint64_t timeout);

	void set_no_delay(bool enabled);

	bool is_throttled() const;
	void set_throttled(bool throttled);
};

}

#endif
