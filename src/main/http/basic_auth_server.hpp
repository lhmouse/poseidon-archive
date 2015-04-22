// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_BASIC_AUTH_SERVER_HPP_
#define POSEIDON_HTTP_BASIC_AUTH_SERVER_HPP_

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "../tcp_server_base.hpp"

namespace Poseidon {

namespace Http {
	class BasicAuthServer : public TcpServerBase {
	public:
		typedef const std::vector<std::string> BasicAuthInfo;

	private:
		const boost::shared_ptr<BasicAuthInfo> m_authInfo;
		const std::string m_path;

	public:
		BasicAuthServer(const IpPort &bindAddr, const char *cert, const char *privateKey,
			std::vector<std::string> authInfo, std::string path);
		~BasicAuthServer();

	protected:
		boost::shared_ptr<TcpSessionBase> onClientConnect(UniqueFile client) const OVERRIDE  = 0;

	public:
		const boost::shared_ptr<BasicAuthInfo> &getAuthInfo() const {
			return m_authInfo;
		}
		const std::string &getPath() const {
			return m_path;
		}
	};
}

}

#endif
