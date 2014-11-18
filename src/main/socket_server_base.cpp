// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "socket_server_base.hpp"
#define POSEIDON_SOCK_ADDR_
#include "sock_addr.hpp"
#include <fcntl.h>
using namespace Poseidon;

SocketServerBase::SocketServerBase(ScopedFile socket)
	: m_socket(STD_MOVE(socket)), m_localInfo(getLocalIpPortFromFd(m_socket.get()))
{
	const int flags = ::fcntl(m_socket.get(), F_GETFL);
	if(flags == -1){
		const int code = errno;
		LOG_POSEIDON_ERROR("Could not get fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}
	if(::fcntl(m_socket.get(), F_SETFL, flags | O_NONBLOCK) != 0){
		const int code = errno;
		LOG_POSEIDON_ERROR("Could not set fcntl flags on socket.");
		DEBUG_THROW(SystemError, code);
	}

	LOG_POSEIDON_INFO("Created socket server, local = ", m_localInfo);
}
SocketServerBase::~SocketServerBase(){
	LOG_POSEIDON_INFO("Destroyed socket server, local = ", m_localInfo);
}
