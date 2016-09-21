// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_client_base.hpp"
#include "tcp_session_base.hpp"
#include "ssl_factories.hpp"
#include "ssl_filter_base.hpp"
#include "sock_addr.hpp"
#include "ip_port.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <openssl/ssl.h>
#include "singletons/epoll_daemon.hpp"
#include "system_exception.hpp"
#include "log.hpp"

namespace Poseidon {

namespace {
	class SslFilter : public SslFilterBase {
	public:
		SslFilter(Move<UniqueSsl> ssl, int fd)
			: SslFilterBase(STD_MOVE(ssl), fd)
		{
			::SSL_set_connect_state(get_ssl());
		}
	};

	UniqueFile create_socket(int family){
		UniqueFile client(::socket(family, SOCK_STREAM, IPPROTO_TCP));
		if(!client){
			DEBUG_THROW(SystemException);
		}
		return client;
	}
}

TcpClientBase::TcpClientBase(const SockAddr &addr, bool use_ssl, bool accept_invalid_cert)
	: SockAddr(addr), TcpSessionBase(create_socket(SockAddr::get_family()))
{
	real_connect(use_ssl, accept_invalid_cert);
}
TcpClientBase::TcpClientBase(const IpPort &addr, bool use_ssl, bool accept_invalid_cert)
	: SockAddr(get_sock_addr_from_ip_port(addr)), TcpSessionBase(create_socket(SockAddr::get_family()))
{
	real_connect(use_ssl, accept_invalid_cert);
}
TcpClientBase::~TcpClientBase(){
}

void TcpClientBase::real_connect(bool use_ssl, bool accept_invalid_cert){
	if(::connect(m_socket.get(), static_cast<const ::sockaddr *>(SockAddr::get_data()), SockAddr::get_size()) != 0){
		if(errno != EINPROGRESS){
			DEBUG_THROW(SystemException);
		}
	}
	if(use_ssl){
		LOG_POSEIDON_INFO("Initiating SSL handshake...");

		m_ssl_factory.reset(new ClientSslFactory(accept_invalid_cert));
		AUTO(ssl, m_ssl_factory->create_ssl());
		boost::scoped_ptr<SslFilterBase> filter(new SslFilter(STD_MOVE(ssl), get_fd()));
		init_ssl(STD_MOVE(filter));
	}
}

void TcpClientBase::go_resident(){
	EpollDaemon::add_session(virtual_shared_from_this<TcpSessionBase>());
}

}
