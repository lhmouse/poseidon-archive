// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "server.hpp"
#include "session.hpp"

namespace Poseidon {

CbppServer::CbppServer(std::size_t category, const IpPort &bindAddr,
	const char *cert, const char *privateKey)
	: TcpServerBase(bindAddr, cert, privateKey)
	, m_category(category)
{
}

boost::shared_ptr<TcpSessionBase> CbppServer::onClientConnect(UniqueFile client) const {
	return boost::make_shared<CbppSession>(m_category, STD_MOVE(client));
}

}
