// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_server_base.hpp"
#include "tcp_session_base.hpp"
#include "ssl_factories.hpp"
#include "ssl_filter.hpp"
#include "sock_addr.hpp"
#include "ip_port.hpp"
#include "singletons/main_config.hpp"
#include "singletons/epoll_daemon.hpp"
#include "log.hpp"
#include "system_exception.hpp"
#include "profiler.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <openssl/ssl.h>

namespace Poseidon {

namespace {
	Unique_file create_tcp_socket(const Sock_addr &addr){
		Unique_file tcp;
		POSEIDON_THROW_UNLESS(tcp.reset(::socket(addr.get_family(), SOCK_STREAM | SOCK_NONBLOCK, IPPROTO_TCP)), System_exception);
		static CONSTEXPR const int s_true_value = true;
		POSEIDON_THROW_UNLESS(::setsockopt(tcp.get(), SOL_SOCKET, SO_REUSEADDR, &s_true_value, sizeof(s_true_value)) == 0, System_exception);
		POSEIDON_THROW_UNLESS(::bind(tcp.get(), static_cast<const ::sockaddr *>(addr.data()), static_cast<unsigned>(addr.size())) == 0, System_exception);
		POSEIDON_THROW_UNLESS(::listen(tcp.get(), SOMAXCONN) == 0, System_exception);
		return tcp;
	}
}

Tcp_server_base::Tcp_server_base(const Sock_addr &addr, const char *certificate, const char *private_key)
	: Socket_base(create_tcp_socket(addr))
{
	if(certificate && *certificate){
		m_ssl_factory.reset(new Ssl_server_factory(certificate, private_key));
	}

	POSEIDON_LOG(Logger::special_major | Logger::level_info, "Created TCP server on ", get_local_info(), ", SSL = ", !!m_ssl_factory);
}
Tcp_server_base::~Tcp_server_base(){
	POSEIDON_LOG(Logger::special_major | Logger::level_info, "Destroyed TCP server on ", get_local_info(), ", SSL = ", !!m_ssl_factory);
}

int Tcp_server_base::poll_read_and_process(unsigned char */*hint_buffer*/, std::size_t /*hint_capacity*/, bool /*readable*/){
	POSEIDON_PROFILE_ME;

	for(unsigned i = 0; i < 16; ++i){
		boost::shared_ptr<Tcp_session_base> session;
		try {
			Unique_file client;
			if(!client.reset(::accept4(get_fd(), NULLPTR, NULLPTR, SOCK_NONBLOCK))){
				return errno;
			}
			session = on_client_connect(STD_MOVE(client));
			if(!session){
				POSEIDON_LOG_WARNING("on_client_connect() returns a null pointer.");
				return EWOULDBLOCK;
			}
		} catch(std::exception &e){
			POSEIDON_LOG_ERROR("std::exception thrown: what = ", e.what());
			return EINTR;
		} catch(...){
			POSEIDON_LOG_ERROR("Unknown exception thrown.");
			return EINTR;
		}
		try {
			if(m_ssl_factory){
				boost::scoped_ptr<Ssl_filter> ssl_filter;
				m_ssl_factory->create_ssl_filter(ssl_filter, session->get_fd());
				session->init_ssl(ssl_filter);
			}
			const AUTO(tcp_request_timeout, Main_config::get<std::uint64_t>("tcp_request_timeout", 5000));
			session->set_timeout(tcp_request_timeout);
			Epoll_daemon::add_socket(session, true);
			POSEIDON_LOG_INFO("Accepted TCP connection from ", session->get_remote_info());
		} catch(std::exception &e){
			POSEIDON_LOG_ERROR("std::exception thrown: what = ", e.what());
			session->force_shutdown();
			continue;
		}
	}
	return 0;
}

}
