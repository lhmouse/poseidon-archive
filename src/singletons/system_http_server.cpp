// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "system_http_server.hpp"
#include "main_config.hpp"
#include "epoll_daemon.hpp"
#include "module_depository.hpp"
#include "profile_depository.hpp"
#include <signal.h>
#include "../log.hpp"
#include "../exception.hpp"
#include "../profiler.hpp"
#include "../tcp_server_base.hpp"
#include "../http/session.hpp"
#include "../http/urlencoded.hpp"
#include "../http/exception.hpp"
#include "../http/authorization.hpp"
#include "../http/url_param.hpp"
#include "../csv_document.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {

namespace {
	class SystemSession : public Http::Session {
	private:
		const boost::shared_ptr<const Http::AuthInfo> m_auth_info;
		const std::string m_prefix;

	public:
		SystemSession(UniqueFile socket, boost::shared_ptr<const Http::AuthInfo> auth_info, std::string prefix)
			: Http::Session(STD_MOVE(socket))
			, m_auth_info(STD_MOVE(auth_info)), m_prefix(STD_MOVE(prefix))
		{
		}

	protected:
		void on_sync_request(Http::RequestHeaders request_headers, StreamBuffer /* entity */) OVERRIDE {
			PROFILE_ME;
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Accepted system HTTP request from ", get_remote_info());

			check_and_throw_if_unauthorized(m_auth_info, get_remote_info(), request_headers);

			try {
				Buffer_istream uri_is;
				uri_is.set_buffer(StreamBuffer(request_headers.uri));
				std::string uri;
				Http::url_decode(uri_is, uri);
				if((uri.size() < m_prefix.size()) || (uri.compare(0, m_prefix.size(), m_prefix) != 0)){
					LOG_POSEIDON_WARNING("Inacceptable system HTTP request: uri = ", request_headers.uri, ", prefix = ", m_prefix);
					DEBUG_THROW(Http::Exception, Http::ST_NOT_FOUND);
				}
				uri.erase(0, m_prefix.size());

				if(request_headers.verb != Http::V_GET){
					DEBUG_THROW(Http::Exception, Http::ST_METHOD_NOT_ALLOWED);
				}

				if(uri == "shutdown"){
					LOG_POSEIDON_WARNING("Received shutdown HTTP request. The server will be shutdown now.");
					::raise(SIGTERM);
					send_default(Http::ST_OK);
				} else if(uri == "load_module"){
					AUTO_REF(name, request_headers.get_params.at("name"));
					if(!ModuleDepository::load_nothrow(name.c_str())){
						LOG_POSEIDON_WARNING("Failed to load module: ", name);
						send_default(Http::ST_GONE);
						return;
					}
					send_default(Http::ST_OK);
				} else if(uri == "unload_module"){
					const AUTO_REF(base_addr_str, request_headers.get_params.at("base_addr"));
					Buffer_istream base_addr_is;
					base_addr_is.set_buffer(StreamBuffer(base_addr_str));
					void *base_addr;
					if(!(base_addr_is >>base_addr)){
						LOG_POSEIDON_WARNING("Bad base_addr string: ", base_addr_str);
						send_default(Http::ST_BAD_REQUEST);
						return;
					}
					if(!ModuleDepository::unload(base_addr)){
						LOG_POSEIDON_WARNING("Module not loaded: base address = ", base_addr);
						send_default(Http::ST_GONE);
						return;
					}
					send_default(Http::ST_OK);
				} else if(uri == "show_profile"){
					CsvDocument csv;
					boost::container::map<SharedNts, std::string> row;
					AUTO(snapshot, ProfileDepository::snapshot());
					for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
						row[sslit("file")] = it->file;
						row[sslit("line")] = boost::lexical_cast<std::string>(it->line);
						row[sslit("func")] = it->func;
						row[sslit("samples")] = boost::lexical_cast<std::string>(it->samples);
						row[sslit("ns_total")] = boost::lexical_cast<std::string>(it->ns_total);
						row[sslit("ns_exclusive")] = boost::lexical_cast<std::string>(it->ns_exclusive);
						if(csv.empty()){
							csv.reset_headers(row);
						}
						csv.append(row);
					}

					OptionalMap headers;
					headers.set(sslit("Content-Type"), "text/csv");
					headers.set(sslit("Content-Disposition"), "attachment; name=\"profile.csv\"");
					send(Http::ST_OK, STD_MOVE(headers), StreamBuffer(csv.dump()));
				} else if(uri == "clear_profile"){
					LOG_POSEIDON_WARNING("Cleaning up profile data...");
					ProfileDepository::clear();
					send_default(Http::ST_OK);
				} else if(uri == "show_modules"){
					CsvDocument csv;
					boost::container::map<SharedNts, std::string> row;
					AUTO(snapshot, ModuleDepository::snapshot());
					for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
						row[sslit("real_path")] = it->real_path;
						row[sslit("base_addr")] = boost::lexical_cast<std::string>(it->base_addr);
						row[sslit("ref_count")] = boost::lexical_cast<std::string>(it->ref_count);
						if(csv.empty()){
							csv.reset_headers(row);
						}
						csv.append(row);
					}

					OptionalMap headers;
					headers.set(sslit("Content-Type"), "text/csv");
					headers.set(sslit("Content-Disposition"), "attachment; name=\"modules.csv\"");
					send(Http::ST_OK, STD_MOVE(headers), StreamBuffer(csv.dump()));
				} else if(uri == "show_connections"){
					CsvDocument csv;
					boost::container::map<SharedNts, std::string> row;
					AUTO(snapshot, EpollDaemon::snapshot());
					for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
						row[sslit("remote_ip")] = it->remote.ip.get();
						row[sslit("remote_port")] = boost::lexical_cast<std::string>(it->remote.port);
						row[sslit("local_ip")] = it->local.ip.get();
						row[sslit("local_port")] = boost::lexical_cast<std::string>(it->local.port);
						row[sslit("ms_onlinet")] = boost::lexical_cast<std::string>(it->ms_online);
						if(csv.empty()){
							csv.reset_headers(row);
						}
						csv.append(row);
					}

					OptionalMap headers;
					headers.set(sslit("Content-Type"), "text/csv");
					headers.set(sslit("Content-Disposition"), "attachment; name=\"modules.csv\"");
					send(Http::ST_OK, STD_MOVE(headers), StreamBuffer(csv.dump()));
				} else if(uri == "set_log_mask"){
					const Http::UrlParam to_disable(STD_MOVE(request_headers.headers), "to_disable");
					const Http::UrlParam to_enable(STD_MOVE(request_headers.headers), "to_enable");
					Logger::set_mask(to_disable.as_unsigned(), to_enable.as_unsigned());
					send_default(Http::ST_OK);
				} else {
					LOG_POSEIDON_WARNING("No system HTTP handler: ", uri);
					DEBUG_THROW(Http::Exception, Http::ST_NOT_FOUND);
				}
			} catch(std::logic_error &){
				DEBUG_THROW(Http::Exception, Http::ST_BAD_REQUEST);
			}
		}
	};

	class SystemServer : public TcpServerBase {
	private:
		const boost::shared_ptr<const Http::AuthInfo> m_auth_info;
		const std::string m_path;

	public:
		SystemServer(const IpPort &bind_addr, const char *cert, const char *private_key,
			std::vector<std::string> user_pass, std::string path)
			: TcpServerBase(bind_addr, cert, private_key)
			, m_auth_info(Http::create_auth_info(STD_MOVE(user_pass))), m_path(STD_MOVE(path))
		{
		}

	public:
		boost::shared_ptr<TcpSessionBase> on_client_connect(UniqueFile client) const OVERRIDE {
			return boost::make_shared<SystemSession>(STD_MOVE(client), m_auth_info, m_path + '/');
		}
	};

	boost::shared_ptr<SystemServer> g_system_server;
}

void SystemHttpServer::start(){
	AUTO(bind, MainConfig::get<std::string>     ("system_http_bind"));
	AUTO(port, MainConfig::get<unsigned>        ("system_http_port"));
	AUTO(cert, MainConfig::get<std::string>     ("system_http_certificate"));
	AUTO(pkey, MainConfig::get<std::string>     ("system_http_private_key"));
	AUTO(auth, MainConfig::get_all<std::string> ("system_http_auth_user_pass"));
	AUTO(path, MainConfig::get<std::string>     ("system_http_path", "/"));

	if(bind.empty()){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "System HTTP server is disabled.");
	} else {
		const IpPort bind_addr(SharedNts(bind), port);
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Initializing system HTTP server on ", bind_addr);
		AUTO(server, boost::make_shared<SystemServer>(bind_addr, cert.c_str(), pkey.c_str(), STD_MOVE(auth), STD_MOVE(path)));
		g_system_server = server;
		EpollDaemon::register_server(STD_MOVE_IDN(server));
	}
}
void SystemHttpServer::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Shutting down system HTTP server...");

	g_system_server.reset();
}

}
