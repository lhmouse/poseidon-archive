// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PLAYER_PROTOCOL_BASE_HPP_
#define POSEIDON_PLAYER_PROTOCOL_BASE_HPP_

#include <string>
#include <vector>
#include <algorithm>
#include "../vint50.hpp"
#include "../stream_buffer.hpp"
#include "exception.hpp"
#include "status.hpp"

#define THROW_END_OF_STREAM_	\
	DEBUG_THROW(::Poseidon::PlayerProtocolException,	\
		::Poseidon::PLAYER_END_OF_STREAM, ::Poseidon::SharedNts::observe("End of stream encountered"))

#define THROW_JUNK_AFTER_PACKET_	\
	DEBUG_THROW(::Poseidon::PlayerProtocolException,	\
		::Poseidon::PLAYER_JUNK_AFTER_PACKET, ::Poseidon::SharedNts::observe("Junk after packet body"))

namespace Poseidon {

struct ProtocolBase {
};

}

#endif
