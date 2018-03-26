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
	Unique_file create_tcp_socket(const Sock_addr &addr){
		Unique_file tcp;
		DEBUG_THROW_UNLESS(tcp.reset(::socket(addr.get_family(), SOCK_STREAM | SOCK_NONBLOCK, IPPROTO_TCP)), System_exception);
		DEBUG_THROW_UNLESS((::connect(tcp.get(), static_cast<const ::sockaddr *>(addr.data()), static_cast<unsigned>(addr.size())) == 0) || (errno == EINPROGRESS), System_exception);
		return tcp;
	}
}

Tcp_client_base::Tcp_client_base(const Sock_addr &addr, bool use_ssl, bool verify_peer)
	: Tcp_session_base(create_tcp_socket(addr))
{
	if(use_ssl){
		LOG_POSEIDON_INFO("Initiating SSL handshake...");
		m_ssl_factory.reset(new Ssl_client_factory(verify_peer));
		boost::scoped_ptr<Ssl_filter> ssl_filter;
		m_ssl_factory->create_ssl_filter(ssl_filter, get_fd());
		Tcp_session_base::init_ssl(ssl_filter);
	}
}
Tcp_client_base::~Tcp_client_base(){
	//
}

}
