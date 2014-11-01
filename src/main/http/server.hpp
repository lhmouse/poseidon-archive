#ifndef POSEIDON_HTTP_SERVER_HPP_
#define POSEIDON_HTTP_SERVER_HPP_

#include "../tcp_server_base.hpp"
#include <set>

namespace Poseidon {

class HttpServer : public TcpServerBase {
private:
	const std::size_t m_category;
	boost::shared_ptr<std::set<std::string> > m_authInfo;

public:
	HttpServer(std::size_t category, std::string bindAddr, unsigned bindPort,
		const char *cert, const char *privateKey, const std::vector<std::string> &authInfo);

protected:
	boost::shared_ptr<class TcpSessionBase> onClientConnect(ScopedFile client) const;
};

}

#endif
