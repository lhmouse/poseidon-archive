#ifndef POSEIDON_PLAYER_EXCEPTION_HPP_
#define POSEIDON_PLAYER_EXCEPTION_HPP_

#include "../exception.hpp"
#include "status.hpp"

namespace Poseidon {

class PlayerException : public ProtocolException {
public:
	PlayerException(const char *file, std::size_t line, PlayerStatus status)
		: ProtocolException(file, line, "Player protocol exception",
			static_cast<unsigned>(status))
	{
	}

public:
	PlayerStatus status() const NOEXCEPT {
		return static_cast<PlayerStatus>(ProtocolException::code());
	}
};

}

#endif
