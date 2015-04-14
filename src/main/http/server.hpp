// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SERVER_HPP_
#define POSEIDON_HTTP_SERVER_HPP_

#include "../tcp_server_base.hpp"
#include <vector>
#include <string>

namespace Poseidon {

class TcpSessionBase;

namespace Http {
	class Server : public TcpServerBase {
	private:
		const std::size_t m_category;
		const boost::shared_ptr<const std::vector<std::string> > m_authInfo;

	public:
		Server(std::size_t category, const IpPort &bindAddr, const char *cert, const char *privateKey,
			std::vector<std::string> authInfo);

	protected:
		const boost::shared_ptr<const std::vector<std::string> > &getAuthInfo() const {
			return m_authInfo;
		}

	protected:
		boost::shared_ptr<TcpSessionBase> onClientConnect(UniqueFile client) const OVERRIDE;

	public:
		std::size_t getCategory() const {
			return m_category;
		}
	};
}

}

#endif
