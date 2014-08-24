#ifndef POSEIDON_TCP_PEER_HPP_
#define POSEIDON_TCP_PEER_HPP_

#include <string>
#include <cstddef>
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include "virtual_shared_from_this.hpp"
#include "raii.hpp"
#include "atomic.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

class TcpPeer : boost::noncopyable
	, public virtual VirtualSharedFromThis
{
private:
	ScopedFile m_socket;
	std::string m_remoteIp;

	volatile bool m_shutdown;
	mutable boost::mutex m_queueMutex;
	StreamBuffer m_sendBuffer;

protected:
	explicit TcpPeer(ScopedFile &socket);
	virtual ~TcpPeer();

public:
	int getFd() const {
		return m_socket.get();
	}
	const std::string &getRemoteIp() const {
		return m_remoteIp;
	}
	bool hasBeenShutdown() const {
		return atomicLoad(m_shutdown);
	}

	// 如果 size 为零则返回所有待发送字节数。
	std::size_t peekWriteAvail(void *data, std::size_t size) const throw();
	void notifyWritten(std::size_t size) throw();

	virtual void onReadAvail(const void *data, std::size_t size) = 0;
	void send(const void *data, std::size_t size);
	void shutdown();
	void forceShutdown();
};

}

#endif
