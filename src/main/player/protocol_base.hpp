#ifndef POSEIDON_PLAYER_PROTOCOL_BASE_HPP_
#define POSEIDON_PLAYER_PROTOCOL_BASE_HPP_

#include <string>
#include <vector>
#include <algorithm>
#include "../vint50.hpp"
#include "../stream_buffer.hpp"
#include "../exception.hpp"

#define THROW_EOS_	\
	DEBUG_THROW(::Poseidon::ProtocolException,	\
		"End of stream encountered.", ::Poseidon::ProtocolException::ERR_END_OF_STREAM)

namespace Poseidon {

struct ProtocolBase {
};

}

#endif
