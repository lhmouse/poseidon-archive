// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_SERVER_BASE_HPP_
#define POSEIDON_SOCKET_SERVER_BASE_HPP_

#include <boost/noncopyable.hpp>
#include "raii.hpp"
#include "ip_port.hpp"

namespace Poseidon {

class SocketServerBase : boost::noncopyable {
private:
	const ScopedFile m_socket;
	const IpPort m_localInfo;

public:
	explicit SocketServerBase(ScopedFile socket);
	virtual ~SocketServerBase();

protected:
	int getFd() const {
		return m_socket.get();
	}

public:
	const IpPort &getLocalInfo() const {
		return m_localInfo;
	}

	virtual bool poll() const = 0;
};

}

#endif
