// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "tcp_client_base.hpp"
#include "ssl_factories.hpp"
#include "ssl_filter.hpp"
#include "sock_addr.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <openssl/ssl.h>
#include "log.hpp"
#include "system_exception.hpp"

namespace Poseidon {

namespace {
	UniqueFile create_tcp_socket(const SockAddr &addr){
		UniqueFile tcp;
		DEBUG_THROW_UNLESS(tcp.reset(::socket(addr.get_family(), SOCK_STREAM | SOCK_NONBLOCK, IPPROTO_TCP)), SystemException);
		DEBUG_THROW_UNLESS((::connect(tcp.get(), static_cast<const ::sockaddr *>(addr.data()), static_cast<unsigned>(addr.size())) == 0) || (errno == EINPROGRESS), SystemException);
		return tcp;
	}
}

TcpClientBase::TcpClientBase(const SockAddr &addr, bool use_ssl, bool verify_peer)
	: TcpSessionBase(create_tcp_socket(addr))
{
	if(use_ssl){
		LOG_POSEIDON_INFO("Initiating SSL handshake...");
		m_ssl_factory.reset(new SslClientFactory(verify_peer));
		boost::scoped_ptr<SslFilter> ssl_filter;
		m_ssl_factory->create_ssl_filter(ssl_filter, get_fd());
		TcpSessionBase::init_ssl(ssl_filter);
	}
}
TcpClientBase::~TcpClientBase(){
	//
}

}
