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

#define THROW_END_OF_STREAM_(protocol_, field_)	\
	DEBUG_THROW(::Poseidon::PlayerProtocolException,	\
		::Poseidon::PLAYER_END_OF_STREAM, ::Poseidon::SharedNts::observe(	\
			"End of stream encountered, expecting "	\
				TOKEN_TO_STR(protocol_) "::" TOKEN_TO_STR(field_) ))

#define THROW_JUNK_AFTER_PACKET_(protocol_)	\
	DEBUG_THROW(::Poseidon::PlayerProtocolException,	\
		::Poseidon::PLAYER_JUNK_AFTER_PACKET, ::Poseidon::SharedNts::observe(	\
			"Junk after packet body, protocol "	\
				TOKEN_TO_STR(protocol_) ))

namespace Poseidon {

struct ProtocolBase {
};

}

#endif
