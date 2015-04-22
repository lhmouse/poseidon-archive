#ifndef POSEIDON_HTTP_BASIC_AUTH_SERVER_HPP_
#define POSEIDON_HTTP_BASIC_AUTH_SERVER_HPP_

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "../tcp_server_base.hpp"

namespace Poseidon {

namespace Http {
	class BasicAuthServer : public TcpServerBase {
	private:
		const boost::shared_ptr<const std::vector<std::string> > m_authInfo;
		const std::string m_path;

	public:
		BasicAuthServer(const IpPort &bindAddr, const char *cert, const char *privateKey,
			std::vector<std::string> authInfo, std::string path);
		~BasicAuthServer();

	protected:
		boost::shared_ptr<TcpSessionBase> onClientConnect(UniqueFile client) const OVERRIDE  = 0;

	public:
		const boost::shared_ptr<const std::vector<std::string> > &getAuthInfo() const {
			return m_authInfo;
		}
		const std::string &getPath() const {
			return m_path;
		}
	};
}

}

#endif
