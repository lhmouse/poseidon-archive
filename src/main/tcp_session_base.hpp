#ifndef POSEIDON_TCP_SESSION_BASE_HPP_
#define POSEIDON_TCP_SESSION_BASE_HPP_

#include "../cxx_ver.hpp"
#include "session_base.hpp"
#include <string>
#include <cstddef>
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/scoped_ptr.hpp>
#include "raii.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

struct TcpSessionImpl;

class TcpSessionBase : public SessionBase {
	friend class TcpSessionImpl;

private:
	class SslImpl;

private:
	ScopedFile m_socket;
	std::string m_remoteIp;
	boost::scoped_ptr<SslImpl> m_ssl;

	volatile bool m_shutdown;
	mutable boost::mutex m_bufferMutex;
	StreamBuffer m_sendBuffer;

protected:
	explicit TcpSessionBase(Move<ScopedFile> socket);
	virtual ~TcpSessionBase();

private:
	long doRead(void *date, unsigned long size);
	long doWrite(boost::mutex::scoped_lock &lock, void *hint, unsigned long hintSize);

protected:
	void initSslClient();
	void initSslServer(const char *certPath, const char *privKeyPath);

public:
	const std::string &getRemoteIp() const;
	void onReadAvail(const void *data, std::size_t size) = 0;
	bool send(StreamBuffer buffer);
	bool hasBeenShutdown() const;
	bool shutdown();
	bool forceShutdown();

	bool shutdown(StreamBuffer buffer);
};

}

#endif
