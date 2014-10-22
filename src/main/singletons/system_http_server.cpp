#include "../../precompiled.hpp"
#include "system_http_server.hpp"
#include <signal.h>
#include "epoll_daemon.hpp"
#include "http_servlet_manager.hpp"
#include "module_manager.hpp"
#include "config_file.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../profiler.hpp"
#include "../http/session.hpp"
#include "../http/exception.hpp"
#include "../shared_ntmbs.hpp"
using namespace Poseidon;

namespace {

void onShutdown(boost::shared_ptr<HttpSession> session, OptionalMap){
	LOG_WARNING("Received shutdown HTTP request. The server will be shutdown now.");
	session->sendDefault(HTTP_OK);
	::raise(SIGTERM);
}

void onLoadModule(boost::shared_ptr<HttpSession> session, OptionalMap getParams){
	AUTO_REF(name, getParams.get("module"));
	if(name.empty()){
		LOG_WARNING("Missing parameter `module`");
		DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
	}
	if(!ModuleManager::loadNoThrow(name)){
		DEBUG_THROW(HttpException, HTTP_NOT_FOUND);
	}
	session->sendDefault(HTTP_OK);
}

void onUnloadModule(boost::shared_ptr<HttpSession> session, OptionalMap getParams){
	AUTO_REF(name, getParams.get("module"));
	if(name.empty()){
		LOG_WARNING("Missing parameter `module`");
		DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
	}
	if(!ModuleManager::unload(name)){
		DEBUG_THROW(HttpException, HTTP_NOT_FOUND);
	}
	session->sendDefault(HTTP_OK);
}

struct JumpTableElement {
	const char *path;
	void (*func)(boost::shared_ptr<HttpSession>, OptionalMap);
};

const JumpTableElement JUMP_TABLE[] = {
	{ "load_module",		&onLoadModule },
	{ "shutdown",			&onShutdown },
	{ "unload_module",		&onUnloadModule },
};

boost::shared_ptr<HttpServer> g_systemServer;
boost::shared_ptr<HttpServlet> g_systemServlet;

void servletProc(boost::shared_ptr<HttpSession> session, HttpRequest request, std::size_t cut){
	LOG_INFO("Accepted system HTTP request from ", session->getRemoteIp());

	if(request.verb != HTTP_GET){
		DEBUG_THROW(HttpException, HTTP_NOT_SUPPORTED);
	}

	if(request.uri.size() < cut){
		DEBUG_THROW(HttpException, HTTP_NOT_FOUND);
	}

	AUTO(lower, BEGIN(JUMP_TABLE));
	AUTO(upper, END(JUMP_TABLE));
	void (*found)(boost::shared_ptr<HttpSession>, OptionalMap) = VAL_INIT;
	do {
		const AUTO(middle, lower + (upper - lower) / 2);
		const int result = std::strcmp(request.uri.c_str() + cut, middle->path);
		if(result == 0){
			found = middle->func;
			break;
		} else if(result < 0){
			upper = middle;
		} else {
			lower = middle + 1;
		}
	} while(lower != upper);

	if(!found){
		LOG_WARNING("No system HTTP handler: ", request.uri);
		DEBUG_THROW(HttpException, HTTP_NOT_FOUND);
	}

	(*found)(STD_MOVE(session), STD_MOVE(request.getParams));
}

}

void SystemHttpServer::start(){
	LOG_INFO("Initializing system HTTP server...");

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

	if(path.empty() || (*path.rbegin() != '/')){
		path.push_back('/');
	}
	g_systemServlet = HttpServletManager::registerServlet(port, path, VAL_INIT,
		TR1::bind(&servletProc, TR1::placeholders::_1, TR1::placeholders::_2, path.size()));

	LOG_INFO("Done initializing system HTTP server.");
}
void SystemHttpServer::stop(){
	LOG_INFO("Shutting down system HTTP server...");

	g_systemServlet.reset();
	g_systemServer.reset();

	LOG_INFO("Done shutting down system HTTP server.");
}
