#include "../../precompiled.hpp"
#include "system_http_server.hpp"
#include <signal.h>
#include "epoll_daemon.hpp"
#include "http_servlet_manager.hpp"
#include "module_manager.hpp"
#include "profile_manager.hpp"
#include "config_file.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../profiler.hpp"
#include "../http/session.hpp"
#include "../http/exception.hpp"
#include "../shared_ntmbs.hpp"
using namespace Poseidon;

namespace {

void escapeCsvField(std::string &dst, const char *src){
	bool needsQuotes = false;
	dst.clear();
	for(;;){
		const char ch = *(src++);
		if(ch == 0){
			break;
		}
		switch(ch){
		case '\"':
			dst.push_back('\"');
		case ',':
		case '\r':
		case '\n':
			needsQuotes = true;
		}
		dst.push_back(ch);
	}
	if(needsQuotes){
		dst.insert(dst.begin(), '\"');
		dst.push_back('\"');
	}
}

void onShutdown(boost::shared_ptr<HttpSession> session, OptionalMap){
	LOG_WARNING("Received shutdown HTTP request. The server will be shutdown now.");
	session->sendDefault(HTTP_OK);
	::raise(SIGTERM);
}

void onLoadModule(boost::shared_ptr<HttpSession> session, OptionalMap getParams){
	AUTO_REF(name, getParams.get("module"));
	if(name.empty()){
		LOG_WARNING("Missing parameter module=");
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
		LOG_WARNING("Missing parameter module=");
		DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
	}
	if(!ModuleManager::unload(name)){
		DEBUG_THROW(HttpException, HTTP_NOT_FOUND);
	}
	session->sendDefault(HTTP_OK);
}

void onProfile(boost::shared_ptr<HttpSession> session, OptionalMap){
	OptionalMap headers;
	headers.set("Content-Type", "text/csv; charset=utf-8");
	headers.set("Content-Disposition", "attachment; name=\"profile.csv\"");

	StreamBuffer contents;
	contents.put("file,line,func,samples,us_total,us_exclusive\r\n");
	AUTO(snapshot, ProfileManager::snapshot());
	std::string str;
	for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
		escapeCsvField(str, it->file.get());
		contents.put(str);
		char temp[256];
		unsigned len = std::sprintf(temp, ",%llu,", (unsigned long long)it->line);
		contents.put(temp, len);
		escapeCsvField(str, it->func.get());
		contents.put(str);
		len = std::sprintf(temp, ",%llu,%llu,%llu\r\n", it->samples, it->usTotal, it->usExclusive);
		contents.put(temp, len);
	}

	session->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

void onModules(boost::shared_ptr<HttpSession> session, OptionalMap){
	OptionalMap headers;
	headers.set("Content-Type", "text/csv; charset=utf-8");
	headers.set("Content-Disposition", "attachment; name=\"modules.csv\"");

	StreamBuffer contents;
	contents.put("path,ref_count\r\n");
	AUTO(snapshot, ModuleManager::snapshot());
	std::string str;
	for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
		escapeCsvField(str, it->path.get());
		contents.put(str);
		char temp[64];
		unsigned len = std::sprintf(temp, ",%llu\r\n", (unsigned long long)it->refCount);
		contents.put(temp, len);
	}

	session->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

void onConnections(boost::shared_ptr<HttpSession> session, OptionalMap){
	OptionalMap headers;
	headers.set("Content-Type", "text/csv; charset=utf-8");
	headers.set("Content-Disposition", "attachment; name=\"connections.csv\"");

	StreamBuffer contents;
	contents.put("remote_ip,remote_port,local_ip,local_port,us_online\r\n");
	AUTO(snapshot, EpollDaemon::snapshot());
	std::string str;
	for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
		escapeCsvField(str, it->remoteIp.get());
		contents.put(str);
		char temp[64];
		unsigned len = std::sprintf(temp, ",%u,", it->remotePort);
		contents.put(temp, len);
		escapeCsvField(str, it->localIp.get());
		contents.put(str);
		len = std::sprintf(temp, ",%u,%llu\r\n", it->localPort, it->usOnline);
		contents.put(temp, len);
	}

	session->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

const std::pair<
	const char *, void (*)(boost::shared_ptr<HttpSession>, OptionalMap)
	> JUMP_TABLE[] =
{
	// 确保字母顺序。
	std::make_pair("connections", &onConnections),
	std::make_pair("load_module", &onLoadModule),
	std::make_pair("modules", &onModules),
	std::make_pair("profile", &onProfile),
	std::make_pair("shutdown", &onShutdown),
	std::make_pair("unload_module", &onUnloadModule),
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
	VALUE_TYPE(JUMP_TABLE[0].second) found = VAL_INIT;
	do {
		const AUTO(middle, lower + (upper - lower) / 2);
		const int result = std::strcmp(request.uri.c_str() + cut, middle->first);
		if(result == 0){
			found = middle->second;
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
