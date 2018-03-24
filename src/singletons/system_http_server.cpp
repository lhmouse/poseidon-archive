// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "system_http_server.hpp"
#include "job_dispatcher.hpp"
#include "main_config.hpp"
#include "epoll_daemon.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../system_http_session.hpp"
#include "../tcp_server_base.hpp"
#include "../system_http_servlet_base.hpp"
#include "../mutex.hpp"
#include "../exception.hpp"
#include "../http/authentication.hpp"

namespace Poseidon {

namespace {
	class SystemSocketServer : public TcpServerBase {
	private:
		const boost::shared_ptr<const Http::AuthenticationContext> m_auth_ctx;

	public:
		SystemSocketServer(const std::string &bind, boost::uint16_t port, const std::string &cert, const std::string &pkey, boost::shared_ptr<const Http::AuthenticationContext> auth_ctx)
			: TcpServerBase(IpPort(bind.c_str(), port), cert.c_str(), pkey.c_str())
			, m_auth_ctx(STD_MOVE(auth_ctx))
		{
			//
		}

	protected:
		virtual boost::shared_ptr<TcpSessionBase> on_client_connect(Move<UniqueFile> client) OVERRIDE {
			PROFILE_ME;

			AUTO(session, boost::make_shared<SystemHttpSession>(STD_MOVE(client), m_auth_ctx));
			LOG_POSEIDON(Logger::special_major | Logger::level_info, "Accepted system TCP client from ", session->get_remote_info());
			session->set_no_delay(true);
			return STD_MOVE_IDN(session);
		}
	};

	boost::shared_ptr<SystemSocketServer> g_server;

	struct UriComparator {
		bool operator()(const char *lhs, const char *rhs) const NOEXCEPT {
			return ::strcasecmp(lhs, rhs) < 0;
		}
	};
	typedef boost::container::flat_map<const char *, boost::weak_ptr<const SystemHttpServletBase>, UriComparator> ServletMap;

	Mutex g_mutex;
	ServletMap g_servlet_map;
}

void SystemHttpServer::start(){
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Starting system HTTP server...");

	const AUTO(bind, MainConfig::get<std::string>("system_http_bind"));
	const AUTO(port, MainConfig::get<boost::uint16_t>("system_http_port"));
	const AUTO(cert, MainConfig::get<std::string>("system_http_certificate"));
	const AUTO(pkey, MainConfig::get<std::string>("system_http_private_key"));
	const AUTO(relm, MainConfig::get<std::string>("system_http_auth_realm", "Poseidon Test Server"));
	const AUTO(auth, MainConfig::get_all<std::string>("system_http_auth_user_pass"));
	if(bind.empty()){
		LOG_POSEIDON(Logger::special_major | Logger::level_info, "System server is disabled.");
	} else {
		const AUTO(auth_ctx, Http::create_authentication_context(relm, auth));
		const AUTO(server, boost::make_shared<SystemSocketServer>(bind, port, cert, pkey, auth_ctx));
		EpollDaemon::add_socket(server, false);
		g_server = server;
	}
}
void SystemHttpServer::stop(){
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Stopping system HTTP server...");

	const Mutex::UniqueLock lock(g_mutex);
	g_servlet_map.clear();
	g_server.reset();
}

boost::shared_ptr<const SystemHttpServletBase> SystemHttpServer::get_servlet(const char *uri){
	PROFILE_ME;

	const Mutex::UniqueLock lock(g_mutex);
	const AUTO(it, g_servlet_map.find(uri));
	if(it == g_servlet_map.end()){
		return VAL_INIT;
	}
	AUTO(servlet, it->second.lock());
	if(!servlet){
		g_servlet_map.erase(it);
		return VAL_INIT;
	}
	return servlet;
}
void SystemHttpServer::get_all_servlets(boost::container::vector<boost::shared_ptr<const SystemHttpServletBase> > &ret){
	PROFILE_ME;

	const Mutex::UniqueLock lock(g_mutex);
	ret.reserve(ret.size() + g_servlet_map.size());
	for(AUTO(it, g_servlet_map.begin()); it != g_servlet_map.end(); ++it){
		AUTO(servlet, it->second.lock());
		if(!servlet){
			continue;
		}
		ret.push_back(STD_MOVE_IDN(servlet));
	}
}

boost::shared_ptr<const SystemHttpServletBase> SystemHttpServer::register_servlet(boost::shared_ptr<SystemHttpServletBase> servlet){
	PROFILE_ME;

	const Mutex::UniqueLock lock(g_mutex);
	const char *const uri = servlet->get_uri();
	DEBUG_THROW_UNLESS(uri[0] == '/', Exception, sslit("System servlet URI must begin with a slash"));
	LOG_POSEIDON_DEBUG("Registering system servlet: uri = ", uri, ", typeid = ", typeid(*servlet).name());
	const AUTO(pair, g_servlet_map.emplace(uri, servlet));
	DEBUG_THROW_UNLESS(pair.second, Exception, sslit("Duplicate system servlet"));
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Registered system servlet: uri = ", uri, ", typeid = ", typeid(*servlet).name());
	return STD_MOVE_IDN(servlet);
}

}
