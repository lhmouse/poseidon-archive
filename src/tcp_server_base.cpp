// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

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
#include "singletons/epoll_daemon.hpp"
#include "log.hpp"
#include "system_exception.hpp"
#include "endian.hpp"

namespace Poseidon {

namespace {
	class SslFilter : public SslFilterBase {
	public:
		SslFilter(Move<UniqueSsl> ssl, int fd)
			: SslFilterBase(STD_MOVE(ssl), fd)
		{
		}

	private:
		bool establish(){
			const int ret = ::SSL_accept(get_ssl());
			if(ret != 1){
				const int err = ::SSL_get_error(get_ssl(), ret);
				if((err == SSL_ERROR_WANT_READ) || (err == SSL_ERROR_WANT_WRITE)){
					return false;
				}
				LOG_POSEIDON_ERROR("::SSL_accept() = ", ret, ", ::SSL_get_error() = ", err);
				DEBUG_THROW(Exception, sslit("::SSL_accept() failed"));
			}
			return true;
		}
	};

	UniqueFile create_listen_socket(const IpPort &addr){
		SockAddr sa = get_sock_addr_from_ip_port(addr);
		UniqueFile listen(::socket(sa.get_family(), SOCK_STREAM, IPPROTO_TCP));
		if(!listen){
			DEBUG_THROW(SystemException);
		}
		const int TRUE_VALUE = true;
		if(::setsockopt(listen.get(), SOL_SOCKET, SO_REUSEADDR, &TRUE_VALUE, sizeof(TRUE_VALUE)) != 0){
			DEBUG_THROW(SystemException);
		}
		if(::bind(listen.get(), static_cast<const ::sockaddr *>(sa.get_data()), sa.get_size())){
			DEBUG_THROW(SystemException);
		}
		if(::listen(listen.get(), SOMAXCONN)){
			DEBUG_THROW(SystemException);
		}
		return listen;
	}
}

TcpServerBase::TcpServerBase(const IpPort &bind_addr, const char *cert, const char *private_key)
	: SocketServerBase(create_listen_socket(bind_addr))
{
	if(cert && (cert[0] != 0)){
		m_ssl_factory.reset(new ServerSslFactory(cert, private_key));
	}

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Created ", (m_ssl_factory ? "SSL " : ""), "TCP server on ", get_local_info());
}
TcpServerBase::~TcpServerBase(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Destroyed ", (m_ssl_factory ? "SSL " : ""), "TCP server on ", get_local_info());
}

bool TcpServerBase::poll() const {
	UniqueFile client(::accept(get_fd(), NULLPTR, NULLPTR));
	if(!client){
		if(errno != EAGAIN){
			DEBUG_THROW(SystemException);
		}
		return false;
	}
	AUTO(session, on_client_connect(STD_MOVE(client)));
	if(!session){
		LOG_POSEIDON_WARNING("on_client_connect() returns a null pointer.");
		DEBUG_THROW(Exception, sslit("Null client pointer"));
	}
	if(m_ssl_factory){
		AUTO(ssl, m_ssl_factory->create_ssl());
		boost::scoped_ptr<SslFilterBase> filter(new SslFilter(STD_MOVE(ssl), session->get_fd()));
		session->init_ssl(STD_MOVE(filter));
	}
	session->set_timeout(EpollDaemon::get_tcp_request_timeout());
	EpollDaemon::add_session(session);
	LOG_POSEIDON_INFO("Accepted TCP connection from ", session->get_remote_info());
	return true;
}

}
