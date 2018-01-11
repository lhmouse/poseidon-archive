// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_client_base.hpp"
#include "ssl_factories.hpp"
#include "ssl_filter.hpp"
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
		m_ssl_factory.reset(new SslClientFactory(verify_peer));
		boost::scoped_ptr<SslFilter> ssl_filter;
		m_ssl_factory->create_ssl_filter(ssl_filter, get_fd());
		TcpSessionBase::init_ssl(ssl_filter);
	}
}
TcpClientBase::~TcpClientBase(){ }

}
