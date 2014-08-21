#include "../precompiled.hpp"
#include "tcp_peer.hpp"
#include "../exceptions.hpp"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <errno.h>
using namespace Poseidon;

TcpPeer::TcpPeer(ScopedFile &socket){
	m_socket.swap(socket);

	unsigned char sa[INET_ADDRSTRLEN | INET6_ADDRSTRLEN];
	::socklen_t salen = sizeof(sa);
	if(::getpeername(m_socket.get(), (::sockaddr *)sa, &salen) != 0){
		THROW(SystemException, errno);
	}
	m_remoteHost.resize(63);
	const char *const text = ::inet_ntop(
		((::sockaddr *)sa)->sa_family, sa, &m_remoteHost[0], m_remoteHost.size()
	);
	if(!text){
		THROW(SystemException, errno);
	}
	m_remoteHost.resize(std::strlen(text));
}

void TcpPeer::send(const void *data, std::size_t size){
	std::size_t total = 0;
	while(total < size){
		const int written = ::send(m_socket.get(),
			(const char *)data + total, std::min<unsigned>(size - total, 0x400u), 0
		);
		if(written < 0){
			THROW(SystemException, errno);
		}
		total += (unsigned)written;
	}
}
