#ifndef POSEIDON_TCP_SESSION_BASE_HPP_
#define POSEIDON_TCP_SESSION_BASE_HPP_

#include "../cxx_ver.hpp"
#include "session_base.hpp"
#include <string>
#include <cstddef>
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include "raii.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

class TcpSessionBase : public SessionBase {
private:
	ScopedFile m_socket;
	std::string m_remoteIp;

	volatile bool m_shutdown;
	mutable boost::mutex m_bufferMutex;
	StreamBuffer m_sendBuffer;

protected:
	explicit TcpSessionBase(Move<ScopedFile> socket);
	virtual ~TcpSessionBase();

public:
	const std::string &getRemoteIp() const;
	void onReadAvail(const void *data, std::size_t size) = 0;
	bool send(StreamBuffer buffer);
	bool hasBeenShutdown() const;
	bool shutdown();
	bool forceShutdown();

	int getFd() const {
		return m_socket.get();
	}
	// 如果 size 为零则返回所有待发送字节数。
	std::size_t peekWriteAvail(boost::mutex::scoped_lock &lock,
		void *data, std::size_t size) const;
	// 从队列中移除指定的字节数。
	void notifyWritten(std::size_t size);

	bool shutdown(StreamBuffer buffer);
};

}

#endif
