#ifndef POSEIDON_TCP_PEER_HPP_
#define POSEIDON_TCP_PEER_HPP_

#include <string>
#include <deque>
#include <cstddef>
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include "raii.hpp"
#include "atomic.hpp"
#include "virtual_shared_from_this.hpp"

namespace Poseidon {

class TcpPeer : boost::noncopyable
	, public virtual VirtualSharedFromThis
{
private:
	ScopedFile m_socket;
	std::string m_remoteIp;

	volatile bool m_shutdown;
	mutable boost::mutex m_queueMutex;
	std::deque<unsigned char> m_sendQueue;

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

	std::size_t peekWriteAvail(void *data, std::size_t size) const;
	void notifyWritten(std::size_t size);

	virtual void onReadAvail(const void *data, std::size_t size) = 0;
	void send(const void *data, std::size_t size);
	void shutdown();
};

}

#endif
