// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "module_raii.hpp"
#include "singletons/module_depository.hpp"
#include "exception.hpp"

namespace Poseidon {

Handle_stack::~Handle_stack(){
	clear();
}

void Handle_stack::push(boost::shared_ptr<const void> handle){
	m_queue.push_back(STD_MOVE(handle));
}
boost::shared_ptr<const void> Handle_stack::pop(){
	assert(!m_queue.empty());

	AUTO(ret, m_queue.back());
	m_queue.pop_back();
	return ret;
}
void Handle_stack::clear() NOEXCEPT {
	while(!m_queue.empty()){
		m_queue.pop_back();
	}
}

Module_raii_base::Module_raii_base(long priority){
	Module_depository::register_module_raii(this, priority);
}
Module_raii_base::~Module_raii_base(){
	Module_depository::unregister_module_raii(this);
}

}
