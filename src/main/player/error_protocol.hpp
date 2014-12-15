// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PLAYER_ERROR_PROTOCOL_HPP_
#define POSEIDON_PLAYER_ERROR_PROTOCOL_HPP_

#include "protocol_base.hpp"

namespace Poseidon {

#define PROTOCOL_NAME	PlayerErrorProtocol
#define PROTOCOL_ID		0
#define PROTOCOL_FIELDS	\
	FIELD_VUINT(protocolId)	\
	FIELD_VUINT(status)	\
	FIELD_STRING(reason)
#include "protocol_generator.hpp"

}

#endif
