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
#include <boost/thread/mutex.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/function.hpp>
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

	template<class>
	friend class TcpMonitoredSession;

	friend TcpServerBase;
	friend TcpClientBase;

private:
	class OnCloseJob;

private:
	const UniqueFile m_socket;
	const unsigned long long m_createdTime;

	mutable struct {
		volatile bool fetched;
		boost::mutex mutex;
		IpPort remote;
		IpPort local;
	} m_peerInfo;

	boost::scoped_ptr<SslFilterBase> m_sslFilter;

	volatile bool m_shutdown;
	mutable boost::mutex m_bufferMutex;
	StreamBuffer m_sendBuffer;
	Epoll *m_epoll;

	mutable boost::mutex m_onCloseMutex;
	std::deque<boost::function<void ()> > m_onCloseQueue;

	mutable boost::mutex m_timerMutex;
	boost::shared_ptr<const TimerItem> m_shutdownTimer;

protected:
	explicit TcpSessionBase(UniqueFile socket);
	~TcpSessionBase();

private:
	// 注意，同一个 session 只能同时分配给一个 epoll！
	void setEpoll(Epoll *epoll) NOEXCEPT;

	void initSsl(Move<boost::scoped_ptr<SslFilterBase> > sslFilter);
	void pumpOnClose() NOEXCEPT;

	// 同步，线程安全。
	void fetchPeerInfo() const;
	// 和 Windows 的 IsDialogMessage() 类似，这个函数读取并在内部调用 onReadAvail() 处理数据。
	// 这里的出参返回读取的数据，一次性读取的字节数不大于 hintSize。如果开启了 SSL，返回明文。
	long syncReadAndProcess(void *hint, unsigned long hintSize);
	// 这里的出参返回写入的数据，一次性写入的字节数不大于 hintSize。如果开启了 SSL，返回明文。
	long syncWrite(boost::mutex::scoped_lock &lock, void *hint, unsigned long hintSize);

protected:
	// 注意，只能在 epoll 线程中调用这些函数。
	void onReadAvail(const void *data, std::size_t size) OVERRIDE = 0;
	void onClose() NOEXCEPT OVERRIDE FINAL;

public:
	int getFd() const {
		return m_socket.get();
	}

	bool send(StreamBuffer buffer, bool fin = false) OVERRIDE FINAL;
	bool hasBeenShutdown() const OVERRIDE FINAL;
	bool forceShutdown() NOEXCEPT OVERRIDE FINAL;

	unsigned long long getCreatedTime() const {
		return m_createdTime;
	}

	const IpPort &getRemoteInfo() const;
	const IpPort &getLocalInfo() const;

	void setOnClose(boost::function<void ()> callback);
	void setTimeout(unsigned long long timeout);
};

}

#endif
