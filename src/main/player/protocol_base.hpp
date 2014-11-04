#ifndef POSEIDON_PLAYER_PROTOCOL_BASE_HPP_
#define POSEIDON_PLAYER_PROTOCOL_BASE_HPP_

#include <string>
#include <vector>
#include <algorithm>
#include "../vint50.hpp"
#include "../stream_buffer.hpp"
#include "exception.hpp"
#include "status.hpp"

#define THROW_EOS_	\
	DEBUG_THROW(::Poseidon::PlayerProtocolException, PLAYER_END_OF_STREAM)

namespace Poseidon {

struct ProtocolBase {
};

}

#endif
