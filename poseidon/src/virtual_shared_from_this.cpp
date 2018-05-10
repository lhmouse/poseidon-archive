// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "virtual_shared_from_this.hpp"
#include "log.hpp"

namespace Poseidon {

void Virtual_shared_from_this::fail_dynamic_cast(const std::type_info &dst_type, const Virtual_shared_from_this *src){
	LOG_POSEIDON_ERROR("dynamic_cast to `", dst_type.name(), "` failed on an object having type `", typeid(*src).name(), "`");
	throw std::bad_cast();
}

// 虚函数定义在这里，我们在共享库中只保存一份 RTTI。
Virtual_shared_from_this::~Virtual_shared_from_this(){
	//
}

}
