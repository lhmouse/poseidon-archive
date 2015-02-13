// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "system_http_server.hpp"
#include "main_config.hpp"
#include "epoll_daemon.hpp"
#include "http_servlet_depository.hpp"
#include "module_depository.hpp"
#include "profile_depository.hpp"
#include "mysql_daemon.hpp"
#include <signal.h>
#include "../log.hpp"
#include "../exception.hpp"
#include "../profiler.hpp"
#include "../http/session.hpp"
#include "../http/exception.hpp"
#include "../http/server.hpp"
#include "../shared_nts.hpp"
using namespace Poseidon;

namespace {

void escapeCsvField(std::string &dst, const char *src){
	bool needsQuotes = false;
	dst.clear();
	const char *read = src;
	for(;;){
		const char ch = *(read++);
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
	LOG_POSEIDON_WARNING("Received shutdown HTTP request. The server will be shutdown now.");
	session->sendDefault(HTTP_OK);
	::raise(SIGTERM);
}

void onLoadModule(boost::shared_ptr<HttpSession> session, OptionalMap getParams){
	AUTO_REF(name, getParams.at("name"));
	if(!ModuleDepository::loadNoThrow(name.c_str())){
		LOG_POSEIDON_WARNING("Failed to load module: ", name);
		session->sendDefault(HTTP_NOT_FOUND);
		return;
	}
	session->sendDefault(HTTP_OK);
}

void onUnloadModule(boost::shared_ptr<HttpSession> session, OptionalMap getParams){
	AUTO_REF(baseAddrStr, getParams.at("base_addr"));
	std::istringstream iss(baseAddrStr);
	void *baseAddr;
	if(!((iss >>baseAddr) && iss.eof())){
		LOG_POSEIDON_WARNING("Bad base_addr string: ", baseAddrStr);
		session->sendDefault(HTTP_BAD_REQUEST);
		return;
	}
	if(!ModuleDepository::unload(baseAddr)){
		LOG_POSEIDON_WARNING("Module not loaded: base address = ", baseAddr);
		session->sendDefault(HTTP_NOT_FOUND);
		return;
	}
	session->sendDefault(HTTP_OK);
}

void onShowProfile(boost::shared_ptr<HttpSession> session, OptionalMap){
	OptionalMap headers;
	headers.set("Content-Type", "text/csv; charset=utf-8");
	headers.set("Content-Disposition", "attachment; name=\"profile.csv\"");

	StreamBuffer contents;
	contents.put("file,line,func,samples,us_total,us_exclusive\r\n");
	AUTO(snapshot, ProfileDepository::snapshot());
	std::string str;
	for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
		escapeCsvField(str, it->file);
		contents.put(str);
		char temp[256];
		unsigned len = (unsigned)std::sprintf(temp, ",%llu,", (unsigned long long)it->line);
		contents.put(temp, len);
		escapeCsvField(str, it->func);
		contents.put(str);
		len = (unsigned)std::sprintf(temp, ",%llu,%llu,%llu\r\n", it->samples, it->usTotal, it->usExclusive);
		contents.put(temp, len);
	}

	session->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

void onShowModules(boost::shared_ptr<HttpSession> session, OptionalMap){
	OptionalMap headers;
	headers.set("Content-Type", "text/csv; charset=utf-8");
	headers.set("Content-Disposition", "attachment; name=\"modules.csv\"");

	StreamBuffer contents;
	contents.put("real_path,base_addr,ref_count\r\n");
	AUTO(snapshot, ModuleDepository::snapshot());
	std::string str;
	for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
		escapeCsvField(str, it->realPath);
		contents.put(str);
		char temp[256];
		unsigned len = (unsigned)std::sprintf(temp, ",%p,%llu\r\n", it->baseAddr, (unsigned long long)it->refCount);
		contents.put(temp, len);
	}

	session->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

void onShowConnections(boost::shared_ptr<HttpSession> session, OptionalMap){
	OptionalMap headers;
	headers.set("Content-Type", "text/csv; charset=utf-8");
	headers.set("Content-Disposition", "attachment; name=\"connections.csv\"");

	StreamBuffer contents;
	contents.put("remote_ip,remote_port,local_ip,local_port,ms_online\r\n");
	AUTO(snapshot, EpollDaemon::snapshot());
	std::string str;
	for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
		escapeCsvField(str, it->remote.ip);
		contents.put(str);
		char temp[256];
		unsigned len = (unsigned)std::sprintf(temp, ",%u,", it->remote.port);
		contents.put(temp, len);
		escapeCsvField(str, it->local.ip);
		contents.put(str);
		len = (unsigned)std::sprintf(temp, ",%u,%llu\r\n", it->local.port, (unsigned long long)it->msOnline);
		contents.put(temp, len);
	}

	session->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

void onSetLogMask(boost::shared_ptr<HttpSession> session, OptionalMap getParams){
	unsigned long long toEnable = 0, toDisable = 0;
	{
		AUTO_REF(val, getParams.get("to_disable"));
		if(!val.empty()){
			toDisable = boost::lexical_cast<unsigned long long>(val);
		}
	}
	{
		AUTO_REF(val, getParams.get("to_enable"));
		if(!val.empty()){
			toEnable = boost::lexical_cast<unsigned long long>(val);
		}
	}
	Logger::setMask(toDisable, toEnable);
	session->sendDefault(HTTP_OK);
}

void onShowMySqlProfile(boost::shared_ptr<HttpSession> session, OptionalMap){
	OptionalMap headers;
	headers.set("Content-Type", "text/csv; charset=utf-8");
	headers.set("Content-Disposition", "attachment; name=\"mysql_threads.csv\"");

	StreamBuffer contents;
	contents.put("thread,table,us_total\r\n");
	AUTO(snapshot, MySqlDaemon::snapshot());
	std::string str;
	for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
		char temp[256];
		unsigned len = (unsigned)std::sprintf(temp, "%u,", it->thread);
		contents.put(temp, len);
		escapeCsvField(str, it->table);
		contents.put(str);
		len = (unsigned)std::sprintf(temp, ",%llu\r\n", it->usTotal);
		contents.put(temp, len);
	}

	session->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

const std::pair<
	const char *, void (*)(boost::shared_ptr<HttpSession>, OptionalMap)
	> JUMP_TABLE[] =
{
	// 确保字母顺序。
	std::make_pair("load_module", &onLoadModule),
	std::make_pair("set_log_mask", &onSetLogMask),
	std::make_pair("show_connections", &onShowConnections),
	std::make_pair("show_modules", &onShowModules),
	std::make_pair("show_mysql_profile", &onShowMySqlProfile),
	std::make_pair("show_profile", &onShowProfile),
	std::make_pair("shutdown", &onShutdown),
	std::make_pair("unload_module", &onUnloadModule),
};

boost::shared_ptr<HttpServer> g_systemServer;
boost::shared_ptr<HttpServlet> g_systemServlet;

void servletProc(boost::shared_ptr<HttpSession> &session, HttpRequest &request, std::size_t cut){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Accepted system HTTP request from ", session->getRemoteInfo());

	if(request.verb != HTTP_GET){
		DEBUG_THROW(HttpException, HTTP_METHOD_NOT_ALLOWED);
	}

	if(request.uri.size() < cut){
		DEBUG_THROW(HttpException, HTTP_NOT_FOUND);
	}

	AUTO(lower, BEGIN(JUMP_TABLE));
	AUTO(upper, END(JUMP_TABLE));
	VALUE_TYPE(JUMP_TABLE[0].second) found = NULLPTR;
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
		LOG_POSEIDON_WARNING("No system HTTP handler: ", request.uri);
		DEBUG_THROW(HttpException, HTTP_NOT_FOUND);
	}

	(*found)(STD_MOVE(session), STD_MOVE(request.getParams));
}

}

void SystemHttpServer::start(){
	AUTO_REF(conf, MainConfig::getConfigFile());

	AUTO(category, conf.get<std::size_t>("system_http_category", 0));
	AUTO(bind, conf.get<std::string>("system_http_bind", "0.0.0.0"));
	AUTO(port, conf.get<boost::uint16_t>("system_http_port", 8900));
	AUTO(certificate, conf.get<std::string>("system_http_certificate", ""));
	AUTO(privateKey, conf.get<std::string>("system_http_private_key", ""));
	AUTO(authUserPasses, conf.getAll<std::string>("system_http_auth_user_pass"));
	AUTO(path, conf.get<std::string>("system_http_path", "~/sys"));

	if(path.empty() || (*path.rbegin() != '/')){
		path.push_back('/');
	}

	const IpPort bindAddr(SharedNts(bind.c_str()), port);
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Initializing system HTTP server on ", bindAddr);
	g_systemServer = boost::make_shared<HttpServer>(
		category, bindAddr, certificate.c_str(), privateKey.c_str(), authUserPasses);
	EpollDaemon::registerServer(g_systemServer);

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Created system HTTP sevlet on ", path);
	g_systemServlet = HttpServletDepository::registerServlet(
		category, SharedNts(path.c_str()), boost::bind(&servletProc, _1, _2, path.size()));

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Done initializing system HTTP server.");
}
void SystemHttpServer::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Shutting down system HTTP server...");

	g_systemServlet.reset();
	g_systemServer.reset();

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Done shutting down system HTTP server.");
}
