// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "socket_server_base.hpp"
#include <fcntl.h>
#include "ip_port.hpp"
#include "log.hpp"
#include "system_exception.hpp"

namespace Poseidon {

SocketServerBase::SocketServerBase(UniqueFile socket)
	: m_socket(STD_MOVE(socket)), m_local_info(get_local_ip_port_from_fd(m_socket.get()))
{
	const int flags = ::fcntl(m_socket.get(), F_GETFL);
	if(flags == -1){
		const int code = errno;
		LOG_POSEIDON_ERROR("Could not get fcntl flags on socket.");
		DEBUG_THROW(SystemException, code);
	}
	if(::fcntl(m_socket.get(), F_SETFL, flags | O_NONBLOCK) != 0){
		const int code = errno;
		LOG_POSEIDON_ERROR("Could not set fcntl flags on socket.");
		DEBUG_THROW(SystemException, code);
	}

	LOG_POSEIDON_INFO("Created socket server, local = ", m_local_info);
}
SocketServerBase::~SocketServerBase(){
	LOG_POSEIDON_INFO("Destroyed socket server, local = ", m_local_info);
}

}
