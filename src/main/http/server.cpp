// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "server.hpp"
#include "session.hpp"
#include "utilities.hpp"

namespace Poseidon {

namespace Http {
	Server::Server(std::size_t category, const IpPort &bindAddr,
		const char *cert, const char *privateKey, const std::vector<std::string> &authInfo)
		: TcpServerBase(bindAddr, cert, privateKey)
		, m_category(category)
	{
		if(!authInfo.empty()){
			boost::make_shared<std::set<std::string> >().swap(m_authInfo);
			for(AUTO(it, authInfo.begin()); it != authInfo.end(); ++it){
				m_authInfo->insert(base64Encode(*it));
			}
		}
	}

	boost::shared_ptr<TcpSessionBase> Server::onClientConnect(UniqueFile client) const {
		AUTO(session, boost::make_shared<Session>(m_category, STD_MOVE(client)));
		session->setAuthInfo(m_authInfo);
		return session;
	}
}

}
