#ifndef POSEIDON_TCP_CLIENT_BASE_HPP_
#define POSEIDON_TCP_CLIENT_BASE_HPP_

#include "tcp_session_base.hpp"
#include <string>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class TcpClientBase
	: public TcpSessionBase
{
public:
	TcpClientBase(const std::string &ip, unsigned port);
};

}

#endif
