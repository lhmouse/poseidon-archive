#include "../precompiled.hpp"
#include "exception.hpp"
using namespace Poseidon;

PlayerProtocolException::PlayerProtocolException(const char *file, std::size_t line,
	PlayerStatus status, std::string reason)
	: ProtocolException(file, line, STD_MOVE(reason), static_cast<unsigned>(status))
{
}
PlayerProtocolException::~PlayerProtocolException() NOEXCEPT {
}
