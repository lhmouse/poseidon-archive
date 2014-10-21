#ifndef POSEIDON_HTTP_SERVER_HPP_
#define POSEIDON_HTTP_SERVER_HPP_

#include "../tcp_server_base.hpp"
#include <set>

namespace Poseidon {

class HttpServer : public TcpServerBase {
private:
	boost::shared_ptr<std::set<std::string> > m_authInfo;

public:
	HttpServer(const std::string &bindAddr, unsigned bindPort,
		const std::string &cert, const std::string &privateKey,
		const std::vector<std::string> &authInfo);

protected:
	boost::shared_ptr<class TcpSessionBase> onClientConnect(Move<ScopedFile> client) const;
};

}

#endif
