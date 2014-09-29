#include "../../precompiled.hpp"
#include "module_manager.hpp"
#include "../module.hpp"
#include <map>
#include <boost/thread/shared_mutex.hpp>
#include "../log.hpp"
#include "config_file.hpp"
using namespace Poseidon;

namespace {

boost::shared_mutex g_mutex;
std::map<std::string, boost::shared_ptr<Module> > g_modules;

}

void ModuleManager::start(){
	LOG_INFO("Loading init modules...");

	const AUTO(modules, ConfigFile::getAll("init_module"));
	for(AUTO(it, modules.begin()); it != modules.end(); ++it){
		load(*it);
	}
}
void ModuleManager::stop(){
	LOG_INFO("Unloading all modules...");

	const boost::unique_lock<boost::shared_mutex> lock(g_mutex);
	g_modules.clear();
}

boost::shared_ptr<Module> ModuleManager::get(const std::string &path){
	const boost::shared_lock<boost::shared_mutex> lock(g_mutex);
	const AUTO(it, g_modules.find(path));
	if(it == g_modules.end()){
		return NULLPTR;
	}
	return it->second;
}
std::vector<ModuleInfo> ModuleManager::getLoadedList(){
	std::vector<ModuleInfo> ret;
	const boost::shared_lock<boost::shared_mutex> lock(g_mutex);
	for(AUTO(it, g_modules.begin()); it != g_modules.end(); ++it){
		ret.push_back(ModuleInfo());
		ModuleInfo &mi = ret.back();
		mi.name = it->first;
		mi.refCount = it->second.use_count();
	}
	return STD_MOVE(ret);
}

boost::shared_ptr<Module> ModuleManager::load(const std::string &path){
	AUTO(module, get(path));
	if(module){
		return module;
	}
	module = Module::load(path.c_str());
	{
		const boost::unique_lock<boost::shared_mutex> lock(g_mutex);
		g_modules[path] = STD_MOVE(module);
	}
	return module;
}
bool ModuleManager::unload(const std::string &path){
	const boost::unique_lock<boost::shared_mutex> lock(g_mutex);
	const AUTO(it, g_modules.find(path));
	if(it == g_modules.end()){
		return false;
	}
	g_modules.erase(it);
	return true;
}
