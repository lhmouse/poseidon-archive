// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PLAYER_EXCEPTION_HPP_
#define POSEIDON_PLAYER_EXCEPTION_HPP_

#include "../exception.hpp"
#include "status.hpp"

namespace Poseidon {

class PlayerProtocolException : public ProtocolException {
public:
	PlayerProtocolException(const char *file, std::size_t line,
		PlayerStatus status, SharedNts message);
	~PlayerProtocolException() NOEXCEPT;

public:
	PlayerStatus status() const NOEXCEPT {
		return static_cast<PlayerStatus>(ProtocolException::code());
	}
};

}

#endif
