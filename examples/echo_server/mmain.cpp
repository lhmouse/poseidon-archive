#include <poseidon/precompiled.hpp>
#include <poseidon/log.hpp>
#include <poseidon/module_raii.hpp>
#include <poseidon/stream_buffer.hpp>
#include <poseidon/tcp_session_base.hpp>
#include <poseidon/tcp_server_base.hpp>
#include <poseidon/singletons/epoll_daemon.hpp>
#include <poseidon/errno.hpp>

// 转发配置。
const char          g_server_bind_addr   [] = "0.0.0.0";
const unsigned      g_server_bind_port      = 16323;
const std::uint64_t g_keep_alive_timeout    = 15000; // 如果两个请求之间的时间超过该时间（以毫秒计），连接将会被关闭。

class Session : public Poseidon::TcpSessionBase {
public:
	explicit Session(Poseidon::UniqueFile socket)
		: Poseidon::TcpSessionBase(std::move(socket))
	{
		get_remote_info(); // 这样后面 get_remote_info() 都不会抛出异常。
	}
	~Session(){
	}

protected:
	void on_connect() override {
		LOG_POSEIDON_INFO("Connection established: remote = ", get_remote_info());
	}
	void on_read_hup() noexcept override {
		LOG_POSEIDON_INFO("Connection read hup: remote = ", get_remote_info());
	}
	void on_close(int err_code) noexcept override {
		LOG_POSEIDON_INFO("Connection error: remote = ", get_remote_info(),
			", err_code = ", err_code, ", desc = ", Poseidon::get_error_desc(err_code));
	}

	void on_read_avail(Poseidon::StreamBuffer data) override {
		LOG_POSEIDON_INFO("Connection read avail: remote = ", get_remote_info(),
			", data = ", data.dump());

		Poseidon::StreamBuffer buffer;
		buffer.put("[Response from echo server] ");
		buffer.splice(data);
		send(std::move(buffer));

		set_timeout(g_keep_alive_timeout);
	}
};

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
		return boost::make_shared<Session>(std::move(socket));
	}
};

MODULE_RAII(handles){
	auto server = boost::make_shared<Server>(Poseidon::IpPort(Poseidon::sslit(g_server_bind_addr), g_server_bind_port));
	Poseidon::EpollDaemon::register_server(server);
	handles.push(std::move(server));
}
