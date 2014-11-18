// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "module_raii.hpp"
#include "singletons/module_manager.hpp"
#include "exception.hpp"
using namespace Poseidon;

ModuleRaiiBase::ModuleRaiiBase(){
	ModuleManager::registerModuleRaii(this);
}
ModuleRaiiBase::~ModuleRaiiBase(){
	ModuleManager::unregisterModuleRaii(this);
}
