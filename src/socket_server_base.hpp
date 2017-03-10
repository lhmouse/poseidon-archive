// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_SERVER_BASE_HPP_
#define POSEIDON_SOCKET_SERVER_BASE_HPP_

#include "cxx_util.hpp"
#include "raii.hpp"
#include "ip_port.hpp"

namespace Poseidon {

class SocketServerBase : NONCOPYABLE {
private:
	const UniqueFile m_socket;
	const IpPort m_local_info;

public:
	explicit SocketServerBase(UniqueFile socket);
	virtual ~SocketServerBase();

public:
	int get_fd() const {
		return m_socket.get();
	}

	const IpPort &get_local_info() const {
		return m_local_info;
	}
	IpPort get_local_info_nothrow() const NOEXCEPT {
		return m_local_info;
	}

	virtual bool poll() const = 0;
};

}

#endif
