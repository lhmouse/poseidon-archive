// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "virtual_shared_from_this.hpp"

namespace Poseidon {

// 虚函数定义在这里，我们在共享库中只保存一份 RTTI。
VirtualSharedFromThis::~VirtualSharedFromThis(){ }

}
