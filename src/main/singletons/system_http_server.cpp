// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "system_http_server.hpp"
#include "main_config.hpp"
#include "epoll_daemon.hpp"
#include "module_depository.hpp"
#include "profile_depository.hpp"
#include "mysql_daemon.hpp"
#include <signal.h>
#include "../log.hpp"
#include "../exception.hpp"
#include "../profiler.hpp"
#include "../tcp_server_base.hpp"
#include "../http/session.hpp"
#include "../http/exception.hpp"
#include "../shared_nts.hpp"

namespace Poseidon {

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

	class SystemSession : public Http::Session {
	private:
		const std::string m_path;

	public:
		SystemSession(UniqueFile socket, boost::shared_ptr<const std::vector<std::string> > authInfo, std::string path)
			: Http::Session(STD_MOVE(socket), STD_MOVE(authInfo))
			, m_path(STD_MOVE(path))
		{
		}

	protected:
		void onRequest(Http::Verb verb, std::string uri, unsigned /* version */ ,
			OptionalMap getParams, OptionalMap /* reqHeaders */, std::string /* reqContents */ ) OVERRIDE
		{
			PROFILE_ME;
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Accepted system HTTP request from ", getRemoteInfo());

			if((uri.size() < m_path.size()) || (uri.compare(0, m_path.size(), m_path) != 0)){
				LOG_POSEIDON_WARNING("Inacceptable system HTTP request: ", uri);
				DEBUG_THROW(Http::Exception, Http::ST_NOT_FOUND);
			}

			if(verb != Http::V_GET){
				DEBUG_THROW(Http::Exception, Http::ST_METHOD_NOT_ALLOWED);
			}

			if(uri.compare(m_path.size(), std::string::npos, "shutdown") == 0){
				LOG_POSEIDON_WARNING("Received shutdown HTTP request. The server will be shutdown now.");
				sendDefault(Http::ST_OK);
				::raise(SIGTERM);
			} else if(uri.compare(m_path.size(), std::string::npos, "load_module") == 0){
				AUTO_REF(name, getParams.at("name"));
				if(!ModuleDepository::loadNoThrow(name.c_str())){
					LOG_POSEIDON_WARNING("Failed to load module: ", name);
					sendDefault(Http::ST_NOT_FOUND);
					return;
				}
				sendDefault(Http::ST_OK);
			} else if(uri.compare(m_path.size(), std::string::npos, "unload_module") == 0){
				AUTO_REF(baseAddrStr, getParams.at("base_addr"));
				std::istringstream iss(baseAddrStr);
				void *baseAddr;
				if(!((iss >>baseAddr) && iss.eof())){
					LOG_POSEIDON_WARNING("Bad base_addr string: ", baseAddrStr);
					sendDefault(Http::ST_BAD_REQUEST);
					return;
				}
				if(!ModuleDepository::unload(baseAddr)){
					LOG_POSEIDON_WARNING("Module not loaded: base address = ", baseAddr);
					sendDefault(Http::ST_NOT_FOUND);
					return;
				}
				sendDefault(Http::ST_OK);
			} else if(uri.compare(m_path.size(), std::string::npos, "show_profile") == 0){
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

				send(Http::ST_OK, STD_MOVE(headers), STD_MOVE(contents));
			} else if(uri.compare(m_path.size(), std::string::npos, "show_modules") == 0){
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

				send(Http::ST_OK, STD_MOVE(headers), STD_MOVE(contents));
			} else if(uri.compare(m_path.size(), std::string::npos, "show_connections") == 0){
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

				send(Http::ST_OK, STD_MOVE(headers), STD_MOVE(contents));
			} else if(uri.compare(m_path.size(), std::string::npos, "set_log_mask") == 0){
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
				sendDefault(Http::ST_OK);
			} else if(uri.compare(m_path.size(), std::string::npos, "show_mysql_profile") == 0){
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

				send(Http::ST_OK, STD_MOVE(headers), STD_MOVE(contents));
			} else {
				LOG_POSEIDON_WARNING("No system HTTP handler: ", uri);
				DEBUG_THROW(Http::Exception, Http::ST_NOT_FOUND);
			}
		}
	};

	class SystemServer : public TcpServerBase {
	private:
		static boost::shared_ptr<std::vector<std::string> > createAuthInfo(std::vector<std::string> userPass){
			if(userPass.empty()){
				return VAL_INIT;
			}
			std::sort(userPass.begin(), userPass.end());
			return boost::make_shared<std::vector<std::string> >(STD_MOVE(userPass));
		}

	private:
		const boost::shared_ptr<const std::vector<std::string> > m_authInfo;
		const std::string m_path;

	public:
		SystemServer(const IpPort &bindAddr, const char *cert, const char *privateKey,
			std::vector<std::string> userPass, std::string path)
			: TcpServerBase(bindAddr, cert, privateKey)
			, m_authInfo(createAuthInfo(STD_MOVE(userPass))), m_path(STD_MOVE(path))
		{
		}

	public:
		boost::shared_ptr<TcpSessionBase> onClientConnect(UniqueFile client) const OVERRIDE {
			return boost::make_shared<SystemSession>(STD_MOVE(client), m_authInfo, m_path);
		}
	};

	boost::shared_ptr<SystemServer> g_systemServer;
}

void SystemHttpServer::start(){
	AUTO_REF(conf, MainConfig::getConfigFile());

	AUTO(bind,			conf.get<std::string>("system_http_bind", "0.0.0.0"));
	AUTO(port,			conf.get<boost::uint16_t>("system_http_port", 8900));
	AUTO(certificate,	conf.get<std::string>("system_http_certificate"));
	AUTO(privateKey,	conf.get<std::string>("system_http_private_key"));
	AUTO(authUserPass,	conf.getAll<std::string>("system_http_auth_user_pass"));
	AUTO(path,			conf.get<std::string>("system_http_path", "~/sys"));

	if(path.empty() || (*path.rbegin() != '/')){
		path.push_back('/');
	}

	const IpPort bindAddr(SharedNts(bind.c_str()), port);
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Initializing system HTTP server on ", bindAddr);
	g_systemServer = boost::make_shared<SystemServer>(
		bindAddr, certificate.c_str(), privateKey.c_str(), STD_MOVE(authUserPass), STD_MOVE(path));
	EpollDaemon::registerServer(g_systemServer);
}
void SystemHttpServer::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Shutting down system HTTP server...");

	g_systemServer.reset();
}

}
