#ifndef POSEIDON_TCP_PEER_HPP_
#define POSEIDON_TCP_PEER_HPP_

#include <string>
#include <cstddef>
#include <boost/noncopyable.hpp>
#include <boost/enable_shared_from_this.hpp>
#include "raii.hpp"

namespace Poseidon {

class TcpPeer : boost::noncopyable
	, public boost::enable_shared_from_this<TcpPeer>
{
private:
	ScopedFile m_socket;
	std::string m_remoteHost;

protected:
	explicit TcpPeer(ScopedFile &socket);

public:
	virtual void onDataAvail(const void *data, std::size_t size) = 0;
	void send(const void *data, std::size_t size);

	int getFd() const {
		return m_socket.get();
	}
	// ip:port
	const std::string &getRemoteHost() const {
		return m_remoteHost;
	}
};

}

#endif
