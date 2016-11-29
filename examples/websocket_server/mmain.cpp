#include <poseidon/precompiled.hpp>
#include <poseidon/log.hpp>
#include <poseidon/module_raii.hpp>
#include <poseidon/stream_buffer.hpp>
#include <poseidon/websocket/session.hpp>
#include <poseidon/websocket/handshake.hpp>
#include <poseidon/websocket/exception.hpp>
#include <poseidon/http/low_level_session.hpp>
#include <poseidon/tcp_server_base.hpp>
#include <poseidon/singletons/epoll_daemon.hpp>

// 转发配置。
const char          g_server_bind_addr   [] = "0.0.0.0";
const unsigned      g_server_bind_port      = 11562;

class HttpSession : public Poseidon::Http::LowLevelSession {
private:
	Poseidon::Http::RequestHeaders m_request_headers;

public:
	explicit HttpSession(Poseidon::UniqueFile socket)
		: Poseidon::Http::LowLevelSession(std::move(socket))
	{
	}
	~HttpSession(){
	}

protected:
	void on_low_level_request_headers(Poseidon::Http::RequestHeaders request_headers, std::uint64_t) override {
		m_request_headers = std::move(request_headers);

		if(m_request_headers.verb != Poseidon::Http::V_GET){
			send_http_default_and_shutdown(Poseidon::Http::ST_NOT_IMPLEMENTED);
			return;
		}
		if(m_request_headers.uri != "/"){
			send_http_default_and_shutdown(Poseidon::Http::ST_NOT_FOUND);
			return;
		}
	}
	void on_low_level_request_entity(std::uint64_t, Poseidon::StreamBuffer) override {
		//
	}
	boost::shared_ptr<Poseidon::Http::UpgradedSessionBase> on_low_level_request_end(std::uint64_t, Poseidon::OptionalMap) override;

public:
	void send_http_default_and_shutdown(unsigned status_code){
		Poseidon::Http::LowLevelSession::send_default(status_code);
		Poseidon::Http::LowLevelSession::shutdown_read();
		Poseidon::Http::LowLevelSession::shutdown_write();
	}
};

class Session : public Poseidon::WebSocket::Session {
public:
	explicit Session(const boost::shared_ptr<HttpSession> &parent)
		: Poseidon::WebSocket::Session(parent)
	{
	}

protected:
	void on_sync_data_message(Poseidon::WebSocket::OpCode opcode, Poseidon::StreamBuffer payload) override {
		const auto parent = boost::dynamic_pointer_cast<HttpSession>(get_parent());
		if(!parent){
			return;
		}
		if(opcode != Poseidon::WebSocket::OP_DATA_TEXT){
			LOG_POSEIDON_WARNING("Not something we accept: opcode = ", opcode);
			DEBUG_THROW(Poseidon::WebSocket::Exception, Poseidon::WebSocket::ST_INACCEPTABLE);
		}
		LOG_POSEIDON_FATAL("Received: ", payload.dump());
		send(Poseidon::WebSocket::OP_DATA_TEXT, std::move(payload));
	}
};

boost::shared_ptr<Poseidon::Http::UpgradedSessionBase> HttpSession::on_low_level_request_end(std::uint64_t, Poseidon::OptionalMap){
	if(::strcasecmp(m_request_headers.headers.get("Upgrade").c_str(), "websocket") != 0){
		send_http_default_and_shutdown(Poseidon::Http::ST_FORBIDDEN);
		return { };
	}
	auto response_headers = Poseidon::WebSocket::make_handshake_response(m_request_headers);
	if(response_headers.status_code != Poseidon::Http::ST_SWITCHING_PROTOCOLS){
		send_http_default_and_shutdown(response_headers.status_code);
		return { };
	}
	Poseidon::Http::LowLevelSession::send(std::move(response_headers), { });
	return boost::make_shared<Session>(virtual_shared_from_this<HttpSession>());
}

class Server : public Poseidon::TcpServerBase {
public:
	explicit Server(const Poseidon::IpPort &bind_addr)
		: Poseidon::TcpServerBase(bind_addr, nullptr, nullptr)
	{
	}
	~Server(){
	}

public:
	boost::shared_ptr<Poseidon::TcpSessionBase> on_client_connect(Poseidon::UniqueFile socket) const override {
		return boost::make_shared<HttpSession>(std::move(socket));
	}
};

MODULE_RAII(handles){
	auto server = boost::make_shared<Server>(Poseidon::IpPort(Poseidon::sslit(g_server_bind_addr), g_server_bind_port));
	Poseidon::EpollDaemon::register_server(server);
	handles.push(std::move(server));
}
