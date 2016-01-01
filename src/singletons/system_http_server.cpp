// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

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
#include "../http/utilities.hpp"
#include "../http/exception.hpp"
#include "../http/authorization.hpp"
#include "../shared_nts.hpp"

namespace Poseidon {

namespace {
	void escape_csv_field(std::string &dst, const char *src){
		bool needs_quotes = false;
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
				needs_quotes = true;
			}
			dst.push_back(ch);
		}
		if(needs_quotes){
			dst.insert(dst.begin(), '\"');
			dst.push_back('\"');
		}
	}

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
		void on_request_headers(Http::RequestHeaders request_headers, std::string transfer_encoding, boost::uint64_t content_length) OVERRIDE {
			check_and_throw_if_unauthorized(m_auth_info, get_remote_info(), request_headers);

			Http::Session::on_request_headers(STD_MOVE(request_headers), STD_MOVE(transfer_encoding), content_length);
		}

		void on_sync_request(Http::RequestHeaders request_headers, StreamBuffer /* entity */) OVERRIDE {
			PROFILE_ME;
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Accepted system HTTP request from ", get_remote_info());

			try {
				AUTO(uri, Http::url_decode(request_headers.uri));
				if((uri.size() < m_prefix.size()) || (uri.compare(0, m_prefix.size(), m_prefix) != 0)){
					LOG_POSEIDON_WARNING("Inacceptable system HTTP request: uri = ", uri, ", prefix = ", m_prefix);
					DEBUG_THROW(Http::Exception, Http::ST_NOT_FOUND);
				}
				uri.erase(0, m_prefix.size());

				if(request_headers.verb != Http::V_GET){
					DEBUG_THROW(Http::Exception, Http::ST_METHOD_NOT_ALLOWED);
				}

				if(uri == "shutdown"){
					LOG_POSEIDON_WARNING("Received shutdown HTTP request. The server will be shutdown now.");
					send_default(Http::ST_OK);
					::raise(SIGTERM);
				} else if(uri == "load_module"){
					AUTO_REF(name, request_headers.get_params.at("name"));
					if(!ModuleDepository::load_nothrow(name.c_str())){
						LOG_POSEIDON_WARNING("Failed to load module: ", name);
						send_default(Http::ST_NOT_FOUND);
						return;
					}
					send_default(Http::ST_OK);
				} else if(uri == "unload_module"){
					AUTO_REF(base_addr_str, request_headers.get_params.at("base_addr"));
					std::istringstream iss(base_addr_str);
					void *base_addr;
					if(!((iss >>base_addr) && iss.eof())){
						LOG_POSEIDON_WARNING("Bad base_addr string: ", base_addr_str);
						send_default(Http::ST_BAD_REQUEST);
						return;
					}
					if(!ModuleDepository::unload(base_addr)){
						LOG_POSEIDON_WARNING("Module not loaded: base address = ", base_addr);
						send_default(Http::ST_NOT_FOUND);
						return;
					}
					send_default(Http::ST_OK);
				} else if(uri == "show_profile"){
					OptionalMap headers;
					headers.set(sslit("Content-Type"), "text/csv; charset=utf-8");
					headers.set(sslit("Content-Disposition"), "attachment; name=\"profile.csv\"");

					StreamBuffer contents;
					contents.put("file,line,func,samples,ns_total,ns_exclusive\r\n");
					AUTO(snapshot, ProfileDepository::snapshot());
					std::string str;
					for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
						escape_csv_field(str, it->file);
						contents.put(str);
						char temp[256];
						unsigned len = (unsigned)std::sprintf(temp, ",%llu,", (unsigned long long)it->line);
						contents.put(temp, len);
						escape_csv_field(str, it->func);
						contents.put(str);
						len = (unsigned)std::sprintf(temp, ",%llu,%llu,%llu\r\n", it->samples, it->ns_total, it->ns_exclusive);
						contents.put(temp, len);
					}

					send(Http::ST_OK, STD_MOVE(headers), STD_MOVE(contents));
				} else if(uri == "show_modules"){
					OptionalMap headers;
					headers.set(sslit("Content-Type"), "text/csv; charset=utf-8");
					headers.set(sslit("Content-Disposition"), "attachment; name=\"modules.csv\"");

					StreamBuffer contents;
					contents.put("real_path,base_addr,ref_count\r\n");
					AUTO(snapshot, ModuleDepository::snapshot());
					std::string str;
					for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
						escape_csv_field(str, it->real_path);
						contents.put(str);
						char temp[256];
						unsigned len = (unsigned)std::sprintf(temp, ",%p,%llu\r\n", it->base_addr, (unsigned long long)it->ref_count);
						contents.put(temp, len);
					}

					send(Http::ST_OK, STD_MOVE(headers), STD_MOVE(contents));
				} else if(uri == "show_connections"){
					OptionalMap headers;
					headers.set(sslit("Content-Type"), "text/csv; charset=utf-8");
					headers.set(sslit("Content-Disposition"), "attachment; name=\"connections.csv\"");

					StreamBuffer contents;
					contents.put("remote_ip,remote_port,local_ip,local_port,ms_online\r\n");
					AUTO(snapshot, EpollDaemon::snapshot());
					std::string str;
					for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
						escape_csv_field(str, it->remote.ip);
						contents.put(str);
						char temp[256];
						unsigned len = (unsigned)std::sprintf(temp, ",%u,", it->remote.port);
						contents.put(temp, len);
						escape_csv_field(str, it->local.ip);
						contents.put(str);
						len = (unsigned)std::sprintf(temp, ",%u,%llu\r\n", it->local.port, (unsigned long long)it->ms_online);
						contents.put(temp, len);
					}

					send(Http::ST_OK, STD_MOVE(headers), STD_MOVE(contents));
				} else if(uri == "set_log_mask"){
					unsigned long long to_enable = 0, to_disable = 0;
					const AUTO_REF(to_disable_str, request_headers.get_params.get("to_disable"));
					if(!to_disable_str.empty()){
						to_disable = boost::lexical_cast<unsigned long long>(to_disable_str);
					}
					const AUTO_REF(to_enable_str, request_headers.get_params.get("to_enable"));
					if(!to_enable_str.empty()){
						to_enable = boost::lexical_cast<unsigned long long>(to_enable_str);
					}
					Logger::set_mask(to_disable, to_enable);
					send_default(Http::ST_OK);
				} else if(uri == "show_my_sql_profile"){
					OptionalMap headers;
					headers.set(sslit("Content-Type"), "text/csv; charset=utf-8");
					headers.set(sslit("Content-Disposition"), "attachment; name=\"mysql_threads.csv\"");

					StreamBuffer contents;
					contents.put("thread,table,ns_total\r\n");
					AUTO(snapshot, MySqlDaemon::snapshot());
					std::string str;
					for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
						char temp[256];
						unsigned len = (unsigned)std::sprintf(temp, "%u,", it->thread);
						contents.put(temp, len);
						escape_csv_field(str, it->table);
						contents.put(str);
						len = (unsigned)std::sprintf(temp, ",%llu\r\n", it->ns_total);
						contents.put(temp, len);
					}

					send(Http::ST_OK, STD_MOVE(headers), STD_MOVE(contents));
				} else {
					LOG_POSEIDON_WARNING("No system HTTP handler: ", uri);
					DEBUG_THROW(Http::Exception, Http::ST_NOT_FOUND);
				}
			} catch(std::out_of_range &){
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
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Systen HTTP server is disabled.");
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
