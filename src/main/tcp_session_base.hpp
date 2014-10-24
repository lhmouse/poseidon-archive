#ifndef POSEIDON_TCP_SESSION_BASE_HPP_
#define POSEIDON_TCP_SESSION_BASE_HPP_

#include "cxx_ver.hpp"
#include "session_base.hpp"
#include <string>
#include <cstddef>
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/scoped_ptr.hpp>
#include "raii.hpp"
#include "stream_buffer.hpp"
#include "shared_ntmbs.hpp"

namespace Poseidon {

class TcpSessionBase : public SessionBase {
	friend class TcpSessionImpl;
	friend class TcpServerBase;
	friend class TcpClientBase;

private:
	class SslImpl;

private:
	const ScopedFile m_socket;
	const unsigned long long m_createdTime;
	const std::pair<SharedNtmbs, unsigned> m_remoteAddr;
	const std::pair<SharedNtmbs, unsigned> m_localAddr;

	boost::scoped_ptr<SslImpl> m_ssl;

	volatile bool m_shutdown;
	mutable boost::mutex m_bufferMutex;
	StreamBuffer m_sendBuffer;

protected:
	explicit TcpSessionBase(Move<ScopedFile> socket);
	virtual ~TcpSessionBase();

private:
	void initSsl(Move<boost::scoped_ptr<SslImpl> > ssl);

	long doRead(void *date, unsigned long size);
	long doWrite(boost::mutex::scoped_lock &lock, void *hint, unsigned long hintSize);

protected:
	void onReadAvail(const void *data, std::size_t size) = 0;

public:
	bool send(StreamBuffer buffer, bool final = false); // final 置 true 则发送完毕后挂断连接。
	bool hasBeenShutdown() const;
	bool forceShutdown();

	unsigned long long getCreatedTime() const {
		return m_createdTime;
	}

	const SharedNtmbs &getRemoteIp() const {
		return m_remoteAddr.first;
	}
	unsigned getRemotePort() const {
		return m_remoteAddr.second;
	}

	const SharedNtmbs &getLocalIp() const {
		return m_localAddr.first;
	}
	unsigned getLocalPort() const {
		return m_localAddr.second;
	}
};

}

#endif
