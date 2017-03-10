#include <poseidon/precompiled.hpp>
#include <poseidon/log.hpp>
#include <poseidon/module_raii.hpp>
#include <poseidon/stream_buffer.hpp>
#include <poseidon/udp_server_base.hpp>
#include <poseidon/singletons/epoll_daemon.hpp>

// 服务端配置。
const char          g_bind    [] = "0.0.0.0";
const unsigned      g_port       = 17350;

class Server : public Poseidon::UdpServerBase {
public:
	explicit Server(const Poseidon::IpPort &ip_port)
		: Poseidon::UdpServerBase(ip_port)
	{
	}

public:
	void on_receive(const Poseidon::SockAddr &sock_addr, Poseidon::StreamBuffer data) const override {
		LOG_POSEIDON_ERROR("Read UDP packet: remote = ", Poseidon::get_ip_port_from_sock_addr(sock_addr), ", data = ", data);
		send(sock_addr, Poseidon::StreamBuffer("hello world!\n"));
	}
};

MODULE_RAII(handles){
	const auto ip_port = Poseidon::IpPort(Poseidon::SharedNts::view(g_bind), g_port);
	auto server = boost::make_shared<Server>(ip_port);
	LOG_POSEIDON_FATAL("UDP server created successfully on ", ip_port);
	Poseidon::EpollDaemon::register_server(server);
	handles.push(std::move(server));
}
