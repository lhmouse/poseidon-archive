// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "message_base.hpp"

namespace Poseidon {
namespace Cbpp {

MessageBase::~MessageBase(){ }

std::ostream &operator<<(std::ostream &os, const MessageBase &rhs){
	rhs.dump_debug(os);
	return os;
}

}
}
