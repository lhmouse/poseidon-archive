// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "server.hpp"
#include "session.hpp"
#include "utilities.hpp"

namespace Poseidon {

namespace {
	boost::shared_ptr<const std::vector<std::string> > forkAuthInfo(std::vector<std::string> authInfo){
		if(authInfo.empty()){
			return VAL_INIT;
		}
		std::sort(authInfo.begin(), authInfo.end());
		return boost::make_shared<const std::vector<std::string> >(STD_MOVE(authInfo));
	}
}

namespace Http {
	Server::Server(std::size_t category, const IpPort &bindAddr, const char *cert, const char *privateKey,
		std::vector<std::string> authInfo)
		: TcpServerBase(bindAddr, cert, privateKey)
		, m_category(category), m_authInfo(forkAuthInfo(STD_MOVE(authInfo)))
	{
	}

	boost::shared_ptr<TcpSessionBase> Server::onClientConnect(UniqueFile client) const {
		AUTO(session, boost::make_shared<Session>(m_category, STD_MOVE(client)));
		session->setAuthInfo(getAuthInfo());
		return session;
	}
}

}
