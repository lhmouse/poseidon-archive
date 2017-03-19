// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_client_base.hpp"
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
#include "profiler.hpp"

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

	UniqueFile create_tcp_socket(int family){
		UniqueFile tcp;
		if(!tcp.reset(::socket(family, SOCK_STREAM, IPPROTO_TCP))){
			DEBUG_THROW(SystemException);
		}
		return tcp;
	}
}

TcpClientBase::TcpClientBase(const SockAddr &addr, bool use_ssl, bool verify_peer)
	: TcpSessionBase(create_tcp_socket(addr.get_family()))
{
	init_connect(addr, use_ssl, verify_peer);
}
TcpClientBase::TcpClientBase(const IpPort &addr, bool use_ssl, bool verify_peer)
	: TcpSessionBase(create_tcp_socket(get_sock_addr_from_ip_port(addr).get_family())) // XXX: Get addr family from IP directly?
{
	init_connect(get_sock_addr_from_ip_port(addr), use_ssl, verify_peer);
}
TcpClientBase::~TcpClientBase(){
}

void TcpClientBase::init_connect(const SockAddr &addr, bool use_ssl, bool verify_peer){
	if((::connect(get_fd(), static_cast<const ::sockaddr *>(addr.data()), addr.size()) != 0) && (errno != EINPROGRESS)){
		DEBUG_THROW(SystemException);
	}
	if(use_ssl){
		LOG_POSEIDON_INFO("Initiating SSL handshake...");
		m_ssl_factory.reset(new ClientSslFactory(verify_peer));
		AUTO(ssl, m_ssl_factory->create_ssl());
		boost::scoped_ptr<SslFilterBase> filter(new SslFilter(STD_MOVE(ssl), get_fd()));
		init_ssl(STD_MOVE(filter));
	}
}

void TcpClientBase::go_resident(){
	EpollDaemon::add_socket(virtual_shared_from_this<SocketBase>());
}

}
