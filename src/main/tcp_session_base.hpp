// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TCP_SESSION_BASE_HPP_
#define POSEIDON_TCP_SESSION_BASE_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "session_base.hpp"
#include <string>
#include <cstddef>
#include <boost/thread/mutex.hpp>
#include <boost/scoped_ptr.hpp>
#include "raii.hpp"
#include "ip_port.hpp"
#include "stream_buffer.hpp"
#include "shared_ntmbs.hpp"

namespace Poseidon {

class Epoll;
class SslFilterBase;

class TcpSessionBase : public SessionBase {
	friend class Epoll;
	friend class TcpServerBase;
	friend class TcpClientBase;

private:
	const UniqueFile m_socket;
	const unsigned long long m_createdTime;

	Epoll *m_epoll;

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

protected:
	explicit TcpSessionBase(UniqueFile socket);
	virtual ~TcpSessionBase();

private:
	void setEpoll(Epoll *epoll);

	void initSsl(Move<boost::scoped_ptr<SslFilterBase> > sslFilter);

	void fetchPeerInfo() const;

protected:
	void onReadAvail(const void *data, std::size_t size) = 0;

public:
	int getFd() const {
		return m_socket.get();
	}
	long syncRead(void *date, unsigned long size);
	long syncWrite(boost::mutex::scoped_lock &lock, void *hint, unsigned long hintSize);

	bool send(StreamBuffer buffer, bool fin = false); // final 置 true 则发送完毕后挂断连接。
	bool hasBeenShutdown() const;
	bool forceShutdown();

	unsigned long long getCreatedTime() const {
		return m_createdTime;
	}

	const IpPort &getRemoteInfo() const;
	const IpPort &getLocalInfo() const;
};

}

#endif
