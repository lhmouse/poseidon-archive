// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "server.hpp"
#include "session.hpp"

namespace Poseidon {

namespace Cbpp {
	Server::Server(std::size_t category, const IpPort &bindAddr,
		const char *cert, const char *privateKey)
		: TcpServerBase(bindAddr, cert, privateKey)
		, m_category(category)
	{
	}

	boost::shared_ptr<TcpSessionBase> Server::onClientConnect(UniqueFile client) const {
		return boost::make_shared<Session>(m_category, STD_MOVE(client));
	}
}

}
