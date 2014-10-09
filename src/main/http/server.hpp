#ifndef POSEIDON_HTTP_SERVER_HPP_
#define POSEIDON_HTTP_SERVER_HPP_

#include "../tcp_server_base.hpp"

namespace Poseidon {

class HttpServer : public TcpServerBase {
public:
	HttpServer(const std::string &bindAddr, unsigned bindPort);

protected:
	boost::shared_ptr<class TcpSessionBase>
		onClientConnect(Move<ScopedFile> client) const;
};

}

#endif
