#include "precompiled.hpp"
#include "virtual_shared_from_this.hpp"
using namespace Poseidon;

// 虚函数定义在这里，我们在共享库中只保存一份 RTTI 信息。
VirtualSharedFromThis::~VirtualSharedFromThis(){
}
