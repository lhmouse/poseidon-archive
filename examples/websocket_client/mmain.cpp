#include <poseidon/precompiled.hpp>
#include <poseidon/log.hpp>
#include <poseidon/module_raii.hpp>
#include <poseidon/stream_buffer.hpp>
#include <poseidon/websocket/client.hpp>
#include <poseidon/websocket/handshake.hpp>
#include <poseidon/websocket/exception.hpp>
#include <poseidon/http/low_level_client.hpp>
#include <poseidon/singletons/timer_daemon.hpp>

// 转发配置。
const char          g_client_connect_addr [] = "127.0.0.1";
const unsigned      g_client_connect_port    = 11562;

class HttpClient : public Poseidon::Http::LowLevelClient {
private:
	std::string m_sec_websocket_key;

	Poseidon::Http::ResponseHeaders m_response_headers;

	boost::shared_ptr<Poseidon::TimerItem> m_timer;

public:
	HttpClient(const Poseidon::SockAddr &sock_addr, bool use_ssl, std::string sec_websocket_key)
		: Poseidon::Http::LowLevelClient(sock_addr, use_ssl)
		, m_sec_websocket_key(std::move(sec_websocket_key))
	{
	}

	void on_low_level_response_headers(Poseidon::Http::ResponseHeaders response_headers, std::uint64_t) override {
		m_response_headers = std::move(response_headers);
	}
	void on_low_level_response_entity(std::uint64_t, Poseidon::StreamBuffer) override {
	}
	boost::shared_ptr<Poseidon::Http::UpgradedClientBase> on_low_level_response_end(std::uint64_t, Poseidon::OptionalMap) override;
};

class Client : public Poseidon::WebSocket::Client {
public:
	explicit Client(const boost::shared_ptr<Poseidon::Http::LowLevelClient> &parent)
		: Poseidon::WebSocket::Client(parent)
	{
	}
	~Client(){
	}

private:
	void on_sync_data_message(Poseidon::WebSocket::OpCode opcode, Poseidon::StreamBuffer payload) override {
		const auto parent = boost::dynamic_pointer_cast<HttpClient>(get_parent());
		if(!parent){
			return;
		}
		if(opcode != Poseidon::WebSocket::OP_DATA_TEXT){
			LOG_POSEIDON_WARNING("Not something we accept: opcode = ", opcode);
			DEBUG_THROW(Poseidon::WebSocket::Exception, Poseidon::WebSocket::ST_INACCEPTABLE);
		}
		LOG_POSEIDON_ERROR("Received: ", payload);
	}
};

boost::shared_ptr<Poseidon::Http::UpgradedClientBase> HttpClient::on_low_level_response_end(std::uint64_t, Poseidon::OptionalMap){
	LOG_POSEIDON_DEBUG("End of HTTP response: remote = ", get_remote_info_nothrow());
	if(!Poseidon::WebSocket::check_handshake_response(m_response_headers, m_sec_websocket_key)){
		LOG_POSEIDON_ERROR("Invalid WebSocket handshake response.");
		force_shutdown();
		return { };
	}

	LOG_POSEIDON_INFO("Upgrading to WebSocket...");
	auto client = boost::make_shared<Client>(virtual_shared_from_this<HttpClient>());
	m_timer = Poseidon::TimerDaemon::register_timer(1000, 1000,
		std::bind([](boost::weak_ptr<Client> weak_client){
			const auto client = weak_client.lock();
			if(!client){
				return;
			}
			auto payload = Poseidon::StreamBuffer("hello world!");
			LOG_POSEIDON_ERROR("Sending: ", payload);
			client->send(Poseidon::WebSocket::OP_DATA_TEXT, std::move(payload));
		}, boost::weak_ptr<Client>(client)));
	return std::move(client);
}

MODULE_RAII(){
	auto request_pair = Poseidon::WebSocket::make_handshake_request("/", { }, g_client_connect_addr);
	const auto sock_addr = Poseidon::get_sock_addr_from_ip_port(Poseidon::IpPort(Poseidon::sslit(g_client_connect_addr), g_client_connect_port));
	auto client = boost::make_shared<HttpClient>(sock_addr, false, std::move(request_pair.second));
	client->go_resident();
	DEBUG_THROW_ASSERT(client->send(std::move(request_pair.first)));
}
