// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_BASE_HPP_
#define POSEIDON_SOCKET_BASE_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "virtual_shared_from_this.hpp"
#include "raii.hpp"
#include "mutex.hpp"
#include "ip_port.hpp"
#include <boost/optional.hpp>
#include <boost/cstdint.hpp>

namespace Poseidon {

class IpPort;

class SocketBase : public virtual VirtualSharedFromThis {
public:
	// 至少一个此对象存活的条件下连接不会由于 RDHUP 而被关掉。
	class DelayedShutdownGuard;

private:
	const UniqueFile m_socket;
	const boost::uint64_t m_creation_time;

	volatile bool m_shutdown_read;
	volatile bool m_shutdown_write;
	volatile bool m_really_shutdown_write;
	volatile bool m_throttled;
	volatile bool m_timed_out;
	volatile std::size_t m_delayed_shutdown_guard_count;

	mutable Mutex m_info_mutex;
	mutable boost::optional<IpPort> m_remote_info;
	mutable boost::optional<IpPort> m_local_info;
	mutable boost::optional<bool> m_ipv6;

public:
	explicit SocketBase(Move<UniqueFile> socket);
	~SocketBase();

private:
	void fetch_remote_info_unlocked() const;
	void fetch_local_info_unlocked() const;

protected:
	bool should_really_shutdown_write() const NOEXCEPT;
	void set_timed_out() NOEXCEPT;

public:
	int get_fd() const {
		return m_socket.get();
	}
	boost::uint64_t get_creation_time() const {
		return m_creation_time;
	}

	bool is_listening() const;

	virtual bool has_been_shutdown_read() const NOEXCEPT;
	virtual bool has_been_shutdown_write() const NOEXCEPT;
	virtual bool shutdown_read() NOEXCEPT;
	virtual bool shutdown_write() NOEXCEPT;
	virtual void mark_shutdown() NOEXCEPT;
	virtual void force_shutdown() NOEXCEPT;

	virtual bool is_throttled() const;
	void set_throttled(bool throttled);

	bool did_time_out() const NOEXCEPT;

	const IpPort &get_remote_info() const NOEXCEPT;
	const IpPort &get_local_info() const NOEXCEPT;
	bool is_using_ipv6() const NOEXCEPT;

	// 返回一个 errno 告诉 epoll 如何处理。
	virtual int poll_read_and_process(unsigned char *hint_buffer, std::size_t hint_capacity, bool readable);
	virtual int poll_write(Mutex::UniqueLock &write_lock, unsigned char *hint_buffer, std::size_t hint_capacity, bool writeable);
	virtual void on_close(int err_code);
};

class SocketBase::DelayedShutdownGuard : NONCOPYABLE {
private:
	const boost::weak_ptr<SocketBase> m_weak;

public:
	explicit DelayedShutdownGuard(boost::weak_ptr<SocketBase> weak);
	~DelayedShutdownGuard();
};

}

#endif
