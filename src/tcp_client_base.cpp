// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

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

#ifdef POSEIDON_CXX11
	UniqueFile
#else
	Move<UniqueFile>
#endif
		create_tcp_socket(int family)
	{
#ifdef POSEIDON_CXX11
		UniqueFile tcp;
#else
		static __thread UniqueFile tcp;
#endif
		DEBUG_THROW_UNLESS(tcp.reset(::socket(family, SOCK_STREAM, IPPROTO_TCP)), SystemException);
		return STD_MOVE(tcp);
	}
}

TcpClientBase::TcpClientBase(const SockAddr &addr, bool use_ssl, bool verify_peer)
	: TcpSessionBase(create_tcp_socket(addr.get_family()))
{
	DEBUG_THROW_UNLESS((::connect(get_fd(), static_cast<const ::sockaddr *>(addr.data()), static_cast<unsigned>(addr.size())) == 0) || (errno == EINPROGRESS), SystemException);
	if(use_ssl){
		LOG_POSEIDON_INFO("Initiating SSL handshake...");
		m_ssl_factory.reset(new ClientSslFactory(verify_peer));
		UniqueSsl ssl;
		m_ssl_factory->create_ssl(ssl);
		boost::scoped_ptr<SslFilterBase> filter;
		filter.reset(new SslFilter(STD_MOVE(ssl), get_fd()));
		init_ssl(STD_MOVE(filter));
	}
}
TcpClientBase::~TcpClientBase(){ }

}
