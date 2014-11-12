#include "../precompiled.hpp"
#include "exception.hpp"
using namespace Poseidon;

PlayerProtocolException::PlayerProtocolException(const char *file, std::size_t line,
	PlayerStatus status, SharedNtmbs message) NOEXCEPT
	: ProtocolException(file, line, STD_MOVE(message), static_cast<unsigned>(status))
{
}
PlayerProtocolException::~PlayerProtocolException() NOEXCEPT {
}
