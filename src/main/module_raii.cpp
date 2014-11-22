// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "module_raii.hpp"
#include "singletons/module_depository.hpp"
#include "exception.hpp"
using namespace Poseidon;

ModuleRaiiBase::ModuleRaiiBase(){
	ModuleDepository::registerModuleRaii(this);
}
ModuleRaiiBase::~ModuleRaiiBase(){
	ModuleDepository::unregisterModuleRaii(this);
}
