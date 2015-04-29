// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

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
#include <boost/function.hpp>
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
	class OnCloseJob;

public:
	// 至少一个此对象存活的条件下连接不会由于 RDHUP 而被关掉。
	class DelayedShutdownGuard : NONCOPYABLE {
	private:
		const boost::shared_ptr<TcpSessionBase> m_session;

	public:
		explicit DelayedShutdownGuard(boost::shared_ptr<TcpSessionBase> session);
		~DelayedShutdownGuard();
	};

private:
	const UniqueFile m_socket;
	const boost::uint64_t m_createdTime;

	mutable struct {
		volatile bool fetched;
		Mutex mutex;
		IpPort remote;
		IpPort local;
	} m_peerInfo;

	boost::scoped_ptr<SslFilterBase> m_sslFilter;

	volatile bool m_shutdown;
	volatile std::size_t m_delayedShutdownGuardCount;
	mutable Mutex m_bufferMutex;
	StreamBuffer m_sendBuffer;
	boost::weak_ptr<const boost::weak_ptr<Epoll> > m_epoll;

	mutable Mutex m_onCloseMutex;
	std::deque<boost::function<void ()> > m_onCloseQueue;

	mutable Mutex m_timerMutex;
	boost::shared_ptr<const TimerItem> m_shutdownTimer;

protected:
	explicit TcpSessionBase(UniqueFile socket);
	~TcpSessionBase();

private:
	void onClose() NOEXCEPT OVERRIDE;
	virtual void onReadHup() NOEXCEPT;

	void initSsl(Move<boost::scoped_ptr<SslFilterBase> > sslFilter);

	void setEpoll(boost::weak_ptr<const boost::weak_ptr<Epoll> > epoll) NOEXCEPT;

	void pumpOnClose() NOEXCEPT;

	// 同步，线程安全。
	void fetchPeerInfo() const;
	// 和 Windows 的 IsDialogMessage() 类似，这个函数读取并在内部调用 onReadAvail() 处理数据。
	// 这里的出参返回读取的数据，一次性读取的字节数不大于 hintSize。如果开启了 SSL，返回明文。
	long syncReadAndProcess(void *hint, unsigned long hintSize);
	// 这里的出参返回写入的数据，一次性写入的字节数不大于 hintSize。如果开启了 SSL，返回明文。
	long syncWrite(Mutex::UniqueLock &lock, void *hint, unsigned long hintSize);

protected:
	// 注意，只能在 epoll 线程中调用这些函数。
	void onReadAvail(const void *data, std::size_t size) OVERRIDE = 0;

public:
	int getFd() const {
		return m_socket.get();
	}

	bool isUsingSsl() const {
		return !!m_sslFilter;
	}

	bool send(StreamBuffer buffer, bool fin = false) FINAL;

	bool hasBeenShutdown() const OVERRIDE;
	bool shutdown() NOEXCEPT;
	bool forceShutdown() NOEXCEPT OVERRIDE;

	boost::uint64_t getCreatedTime() const {
		return m_createdTime;
	}

	const IpPort &getRemoteInfo() const;
	const IpPort &getLocalInfo() const;

	void registerOnClose(boost::function<void ()> callback);
	void setTimeout(boost::uint64_t timeout);
};

}

#endif
