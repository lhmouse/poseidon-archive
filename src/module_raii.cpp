// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "module_raii.hpp"
#include "singletons/module_depository.hpp"
#include "exception.hpp"

namespace Poseidon {

HandleStack::~HandleStack(){
	clear();
}

void HandleStack::push(boost::shared_ptr<const void> handle){
	m_queue.push_back(STD_MOVE(handle));
}
boost::shared_ptr<const void> HandleStack::pop(){
	assert(!m_queue.empty());

	AUTO(ret, m_queue.back());
	m_queue.pop_back();
	return ret;
}
void HandleStack::clear() NOEXCEPT {
	while(!m_queue.empty()){
		m_queue.pop_back();
	}
}

void HandleStack::swap(HandleStack &rhs) NOEXCEPT {
	m_queue.swap(rhs.m_queue);
}

ModuleRaiiBase::ModuleRaiiBase(long priority){
	ModuleDepository::register_module_raii(this, priority);
}
ModuleRaiiBase::~ModuleRaiiBase(){
	ModuleDepository::unregister_module_raii(this);
}

}
