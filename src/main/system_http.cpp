#include "../precompiled.hpp"
#include "singletons/epoll_daemon.hpp"
#include "singletons/http_servlet_manager.hpp"
#include "singletons/config_file.hpp"
#include "log.hpp"
#include "exception.hpp"
#include "profiler.hpp"
#include "http/session.hpp"
using namespace Poseidon;

namespace {

boost::shared_ptr<HttpServer> g_systemServer;
boost::shared_ptr<HttpServlet> g_systemServlet;

void servletProc(boost::shared_ptr<HttpSession> session, HttpRequest request){
}

}

namespace Poseidon {

void initSystemHttpInterface(){
	LOG_INFO("Initializing system HTTP interface...");

	std::string bind("127.0.0.1");
	boost::uint16_t port = 0;
	std::string certificate;
	std::string privateKey;
	std::vector<std::string> authUserPasses;
	std::string path("/~sys");

	ConfigFile::get(bind, "system_http_bind");
	ConfigFile::get(port, "system_http_port");
	ConfigFile::get(certificate, "system_http_certificate");
	ConfigFile::get(privateKey, "system_http_private_key");
	ConfigFile::getAll(authUserPasses, "system_http_auth_user_pass");
	ConfigFile::get(path, "system_http_path");

	g_systemServer = EpollDaemon::registerHttpServer(bind, port, certificate, privateKey, authUserPasses);
	g_systemServlet = HttpServletManager::registerServlet(bind, VAL_INIT, &servletProc);
}

}