#include "../../precompiled.hpp"
#include "server.hpp"
#include "session.hpp"
#include "../singletons/http_servlet_manager.hpp"
using namespace Poseidon;

HttpServer::HttpServer(const std::string &bindAddr, unsigned bindPort,
	const std::string &cert, const std::string &privateKey)
	: TcpServerBase(bindAddr, bindPort, cert, privateKey)
{
}

boost::shared_ptr<TcpSessionBase>
	HttpServer::onClientConnect(Move<ScopedFile> client) const
{
	AUTO(session, boost::make_shared<HttpSession>(STD_MOVE(client)));
	session->setRequestTimeout(HttpServletManager::getRequestTimeout());
	return session;
}
