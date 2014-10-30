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
