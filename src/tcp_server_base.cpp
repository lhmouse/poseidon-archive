// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_server_base.hpp"
#include "tcp_session_base.hpp"
#include "ssl_factories.hpp"
#include "ssl_filter_base.hpp"
#include "sock_addr.hpp"
#include "ip_port.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <openssl/ssl.h>
#include "singletons/main_config.hpp"
#include "singletons/epoll_daemon.hpp"
#include "log.hpp"
#include "system_exception.hpp"
#include "profiler.hpp"

namespace Poseidon {

namespace {
	class SslFilter : public SslFilterBase {
	public:
		SslFilter(Move<UniqueSsl> ssl, int fd)
			: SslFilterBase(STD_MOVE(ssl), fd)
		{
			::SSL_set_accept_state(get_ssl());
		}
	};

#ifdef POSEIDON_CXX11
	UniqueFile
#else
	Move<UniqueFile>
#endif
		create_tcp_socket(const SockAddr &addr)
	{
#ifdef POSEIDON_CXX11
		UniqueFile tcp;
#else
		static __thread UniqueFile tcp;
#endif
		if(!tcp.reset(::socket(addr.get_family(), SOCK_STREAM, IPPROTO_TCP))){
			DEBUG_THROW(SystemException);
		}
		static CONSTEXPR const int TRUE_VALUE = true;
		if(::setsockopt(tcp.get(), SOL_SOCKET, SO_REUSEADDR, &TRUE_VALUE, sizeof(TRUE_VALUE)) != 0){
			DEBUG_THROW(SystemException);
		}
		if(::bind(tcp.get(), static_cast<const ::sockaddr *>(addr.data()), addr.size()) != 0){
			DEBUG_THROW(SystemException);
		}
		if(::listen(tcp.get(), SOMAXCONN) != 0){
			DEBUG_THROW(SystemException);
		}
		return STD_MOVE(tcp);
	}
}

TcpServerBase::TcpServerBase(const SockAddr &addr, const char *certificate, const char *private_key)
	: SocketBase(create_tcp_socket(addr))
{
	if(certificate && *certificate){
		m_ssl_factory.reset(new ServerSslFactory(certificate, private_key));
	}

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
		"Created TCP server on ", get_local_info(), ", SSL = ", !!m_ssl_factory);
}
TcpServerBase::~TcpServerBase(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
		"Destroyed TCP server on ", get_local_info(), ", SSL = ", !!m_ssl_factory);
}

int TcpServerBase::poll_read_and_process(bool readable){
	PROFILE_ME;

	(void)readable;

	for(unsigned i = 0; i < 16; ++i){
		boost::shared_ptr<TcpSessionBase> session;
		try {
			UniqueFile client;
			if(!client.reset(::accept(get_fd(), NULLPTR, NULLPTR))){
				return errno;
			}
			session = on_client_connect(STD_MOVE(client));
			if(!session){
				LOG_POSEIDON_WARNING("on_client_connect() returns a null pointer.");
				return EWOULDBLOCK;
			}
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			return EINTR;
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown.");
			return EINTR;
		}
		try {
			if(m_ssl_factory){
				UniqueSsl ssl;
				m_ssl_factory->create_ssl(ssl);
				boost::scoped_ptr<SslFilterBase> filter;
				filter.reset(new SslFilter(STD_MOVE(ssl), session->get_fd()));
				session->init_ssl(STD_MOVE(filter));
			}
			const AUTO(tcp_request_timeout, MainConfig::get<boost::uint64_t>("tcp_request_timeout", 5000));
			session->set_timeout(tcp_request_timeout);
			EpollDaemon::add_socket(session, true);
			LOG_POSEIDON_INFO("Accepted TCP connection from ", session->get_remote_info());
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			session->force_shutdown();
			continue;
		}
	}
	return 0;
}

}
