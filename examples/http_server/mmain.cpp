#include <poseidon/precompiled.hpp>
#include <poseidon/log.hpp>
#include <poseidon/module_raii.hpp>
#include <poseidon/stream_buffer.hpp>
#include <poseidon/optional_map.hpp>
#include <poseidon/http/session.hpp>
#include <poseidon/http/verbs.hpp>
#include <poseidon/http/status_codes.hpp>
#include <poseidon/http/header_option.hpp>
#include <poseidon/tcp_server_base.hpp>
#include <poseidon/singletons/epoll_daemon.hpp>
#include <poseidon/buffer_streams.hpp>

// 服务端配置。
const char          g_bind    [] = "0.0.0.0";
const unsigned      g_port       = 17550;

class Session : public Poseidon::Http::Session {
public:
	explicit Session(Poseidon::UniqueFile socket)
		: Poseidon::Http::Session(std::move(socket), 0x1000)
	{
	}
	~Session(){
	}

protected:
	void on_sync_request(Poseidon::Http::RequestHeaders request_headers, Poseidon::StreamBuffer entity){
		Poseidon::Buffer_ostream os;
		os <<"Request from " <<get_remote_info().ip <<":" <<std::endl;
		os <<"-----" <<std::endl;
		os <<"Verb    : " <<Poseidon::Http::get_string_from_verb(request_headers.verb) <<std::endl;
		os <<"URI     : " <<request_headers.uri <<std::endl;
		os <<"Version : " <<(request_headers.version / 10000) <<"." <<(request_headers.version % 10000) <<std::endl;
		os <<std::endl;
		os <<"GET params :" <<std::endl;
		for(auto it = request_headers.get_params.begin(); it != request_headers.get_params.end(); ++it){
			os <<"  " <<it->first <<" = " <<it->second <<std::endl;
		}
		os <<"HTTP headers :" <<std::endl;
		for(auto it = request_headers.headers.begin(); it != request_headers.headers.end(); ++it){
			os <<"  " <<it->first <<" : " <<it->second <<std::endl;
		}
		os <<"Entity :" <<std::endl;
		os <<entity <<std::endl;
		os <<"-----" <<std::endl;

		send(Poseidon::Http::ST_OK, std::move(os.get_buffer()), Poseidon::Http::HeaderOption("text/plain", { }));
	}
};

class Server : public Poseidon::TcpServerBase {
public:
	explicit Server(const Poseidon::IpPort &ip_port)
		: Poseidon::TcpServerBase(ip_port, { }, { })
	{
	}

public:
	boost::shared_ptr<Poseidon::TcpSessionBase> on_client_connect(Poseidon::UniqueFile client) const override {
		return boost::make_shared<Session>(std::move(client));
	}
};

MODULE_RAII(handles){
	const auto ip_port = Poseidon::IpPort(Poseidon::SharedNts::view(g_bind), g_port);
	auto server = boost::make_shared<Server>(ip_port);
	LOG_POSEIDON_FATAL("HTTP server created successfully on ", ip_port);
	Poseidon::EpollDaemon::register_server(server);
	handles.push(std::move(server));
}
