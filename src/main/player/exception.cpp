// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

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
