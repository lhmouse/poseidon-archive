#include "../../precompiled.hpp"
#include "server.hpp"
#include "session.hpp"
#include "utilities.hpp"
#include "../singletons/http_servlet_manager.hpp"
using namespace Poseidon;

HttpServer::HttpServer(const std::string &bindAddr, unsigned bindPort,
	const std::string &cert, const std::string &privateKey,
	const std::vector<std::string> &authInfo)
	: TcpServerBase(bindAddr, bindPort, cert, privateKey)
{
	if(!authInfo.empty()){
		boost::make_shared<std::set<std::string> >().swap(m_authInfo);
		for(AUTO(it, authInfo.begin()); it != authInfo.end(); ++it){
			m_authInfo->insert(base64Encode(*it));
		}
	}
}

boost::shared_ptr<TcpSessionBase> HttpServer::onClientConnect(Move<ScopedFile> client) const {
	AUTO(session, boost::make_shared<HttpSession>(STD_MOVE(client)));
	session->setRequestTimeout(HttpServletManager::getRequestTimeout());
	session->setAuthInfo(m_authInfo);
	return session;
}
