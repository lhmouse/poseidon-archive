#ifndef POSEIDON_UDP_SERVER_BASE_HPP_
#define POSEIDON_UDP_SERVER_BASE_HPP_

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "socket_server_base.hpp"

namespace Poseidon {

class UdpServerBase : public SocketServerBase {
public:
	explicit UdpServerBase(const IpPort &bindAddr);
	virtual ~UdpServerBase();

public:
	bool poll() const;
};

}

#endif
