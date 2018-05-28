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
#include "../exception.hpp"
#include "../http/authentication.hpp"

namespace Poseidon {

namespace {
	class System_socket_server : public Tcp_server_base {
	private:
		const boost::shared_ptr<const Http::Authentication_context> m_auth_ctx;

	public:
		System_socket_server(const std::string &bind, std::uint16_t port, const std::string &cert, const std::string &pkey, boost::shared_ptr<const Http::Authentication_context> auth_ctx)
			: Tcp_server_base(Ip_port(bind.c_str(), port), cert.c_str(), pkey.c_str())
			, m_auth_ctx(STD_MOVE(auth_ctx))
		{
			//
		}

	protected:
		virtual boost::shared_ptr<Tcp_session_base> on_client_connect(Move<Unique_file> client) OVERRIDE {
			POSEIDON_PROFILE_ME;

			AUTO(session, boost::make_shared<System_http_session>(STD_MOVE(client), m_auth_ctx));
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "Accepted system TCP client from ", session->get_remote_info());
			session->set_no_delay(true);
			return STD_MOVE_IDN(session);
		}
	};

	boost::shared_ptr<System_socket_server> g_server;

	struct Uri_comparator {
		bool operator()(const char *lhs, const char *rhs) const NOEXCEPT {
			return ::strcasecmp(lhs, rhs) < 0;
		}
	};
	typedef boost::container::flat_map<const char *, boost::weak_ptr<const System_http_servlet_base>, Uri_comparator> Servlet_map;

	std::mutex g_mutex;
	Servlet_map g_servlet_map;
}

void System_http_server::start(){
	POSEIDON_LOG(Logger::special_major | Logger::level_info, "Starting system HTTP server...");

	const AUTO(bind, Main_config::get<std::string>("system_http_bind"));
	const AUTO(port, Main_config::get<std::uint16_t>("system_http_port"));
	const AUTO(cert, Main_config::get<std::string>("system_http_certificate"));
	const AUTO(pkey, Main_config::get<std::string>("system_http_private_key"));
	const AUTO(relm, Main_config::get<std::string>("system_http_auth_realm", "Poseidon Test Server"));
	const AUTO(auth, Main_config::get_all<std::string>("system_http_auth_user_pass"));
	if(bind.empty()){
		POSEIDON_LOG(Logger::special_major | Logger::level_info, "System server is disabled.");
	} else {
		const AUTO(auth_ctx, Http::create_authentication_context(relm, auth));
		const AUTO(server, boost::make_shared<System_socket_server>(bind, port, cert, pkey, auth_ctx));
		Epoll_daemon::add_socket(server, false);
		g_server = server;
	}
}
void System_http_server::stop(){
	POSEIDON_LOG(Logger::special_major | Logger::level_info, "Stopping system HTTP server...");

	const std::lock_guard<std::mutex> lock(g_mutex);
	g_servlet_map.clear();
	g_server.reset();
}

boost::shared_ptr<const System_http_servlet_base> System_http_server::get_servlet(const char *uri){
	POSEIDON_PROFILE_ME;

	const std::lock_guard<std::mutex> lock(g_mutex);
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
void System_http_server::get_all_servlets(boost::container::vector<boost::shared_ptr<const System_http_servlet_base> > &ret){
	POSEIDON_PROFILE_ME;

	const std::lock_guard<std::mutex> lock(g_mutex);
	ret.reserve(ret.size() + g_servlet_map.size());
	for(AUTO(it, g_servlet_map.begin()); it != g_servlet_map.end(); ++it){
		AUTO(servlet, it->second.lock());
		if(!servlet){
			continue;
		}
		ret.push_back(STD_MOVE_IDN(servlet));
	}
}

boost::shared_ptr<const System_http_servlet_base> System_http_server::register_servlet(boost::shared_ptr<System_http_servlet_base> servlet){
	POSEIDON_PROFILE_ME;

	const std::lock_guard<std::mutex> lock(g_mutex);
	const char *const uri = servlet->get_uri();
	POSEIDON_THROW_UNLESS(uri[0] == '/', Exception, Rcnts::view("System servlet URI must begin with a slash"));
	POSEIDON_LOG_DEBUG("Registering system servlet: uri = ", uri, ", typeid = ", typeid(*servlet).name());
	const AUTO(pair, g_servlet_map.emplace(uri, servlet));
	POSEIDON_THROW_UNLESS(pair.second, Exception, Rcnts::view("Duplicate system servlet"));
	POSEIDON_LOG(Logger::special_major | Logger::level_info, "Registered system servlet: uri = ", uri, ", typeid = ", typeid(*servlet).name());
	return STD_MOVE_IDN(servlet);
}

}
