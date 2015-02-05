// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PLAYER_MESSAGE_BASE_HPP_
#define POSEIDON_PLAYER_MESSAGE_BASE_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <boost/cstdint.hpp>
#include "../vint50.hpp"
#include "../stream_buffer.hpp"
#include "exception.hpp"
#include "status.hpp"

#define THROW_END_OF_STREAM_(message_, field_)	\
	DEBUG_THROW(::Poseidon::PlayerMessageException,	\
		::Poseidon::PLAYER_END_OF_STREAM, ::Poseidon::SharedNts::observe(	\
			"End of stream encountered, expecting "	\
				TOKEN_TO_STR(message_) "::" TOKEN_TO_STR(field_) ))

#define THROW_JUNK_AFTER_PACKET_(message_)	\
	DEBUG_THROW(::Poseidon::PlayerMessageException,	\
		::Poseidon::PLAYER_JUNK_AFTER_PACKET, ::Poseidon::SharedNts::observe(	\
			"Junk after packet body, message "	\
				TOKEN_TO_STR(message_) ))

namespace Poseidon {

struct PlayerMessageBase {
	static void encodeHeader(
		StreamBuffer &dst, boost::uint16_t messageId, boost::uint64_t messageLen);
	static bool decodeHeader( // 如果返回 false，不从 src 中消耗任何数据。
		boost::uint16_t &messageId, boost::uint64_t &messageLen, StreamBuffer &src) NOEXCEPT;
};

}

#endif
