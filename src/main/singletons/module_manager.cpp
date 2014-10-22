#include "../../precompiled.hpp"
#include "module_manager.hpp"
#include <map>
#include <boost/thread/mutex.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/type_traits/decay.hpp>
#include <dlfcn.h>
#include "config_file.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../exception.hpp"
using namespace Poseidon;

namespace {

struct ModuleItem {
	boost::shared_ptr<Module> module;
	boost::shared_ptr<const void> context;
};

boost::mutex g_dlMutex;
boost::shared_mutex g_mutex;
std::map<std::string, ModuleItem> g_modules;

struct DynamicLibraryCloser {
	CONSTEXPR void *operator()() NOEXCEPT {
		return VAL_INIT;
	}
	void operator()(void *handle) NOEXCEPT {
		const boost::mutex::scoped_lock lock(g_dlMutex);
		if(::dlclose(handle) != 0){
			LOG_WARNING("Error unloading dynamic library: ", ::dlerror());
		}
	}
};

}

class Poseidon::Module : boost::noncopyable
	, public boost::enable_shared_from_this<Module> {
private:
	std::string m_path;
	ScopedHandle<DynamicLibraryCloser> m_handle;

public:
	explicit Module(std::string path)
		: m_path(STD_MOVE(path))
	{
		LOG_INFO("Loading module: ", m_path);

		const boost::mutex::scoped_lock lock(g_dlMutex);
		m_handle.reset(::dlopen(m_path.c_str(), RTLD_NOW));
		if(!m_handle){
			const char *const error = ::dlerror();
			LOG_ERROR("Error loading dynamic library: ", error);
			DEBUG_THROW(Exception, error);
		}
		LOG_DEBUG("Handle = ", m_handle.get());
	}
	~Module(){
		LOG_INFO("Unloading module: ", m_path);
	}

public:
	void init(boost::shared_ptr<const void> &context){
		void *procSymAddr;
		{
			const boost::mutex::scoped_lock lock(g_dlMutex);
			procSymAddr = ::dlsym(m_handle.get(), "poseidonModuleInit");
			if(!procSymAddr){
				const char *const error = ::dlerror();
				LOG_ERROR("Error getting init function: ", error);
				DEBUG_THROW(Exception, error);
			}
		}

		LOG_INFO("Initializing module: ", m_path);
		const AUTO(initProc, reinterpret_cast<VALUE_TYPE(poseidonModuleInit)>(procSymAddr));
		(*initProc)(shared_from_this(), context);
		LOG_INFO("Done initializing module: ", m_path);
	}
};

void ModuleManager::start(){
	LOG_INFO("Loading init modules...");

	const AUTO(modules, ConfigFile::getAll("init_module"));
	for(AUTO(it, modules.begin()); it != modules.end(); ++it){
		load(*it);
	}
}
void ModuleManager::stop(){
	LOG_INFO("Unloading all modules...");

	g_modules.clear();
}

boost::shared_ptr<Module> ModuleManager::get(const std::string &path){
	const boost::shared_lock<boost::shared_mutex> lock(g_mutex);
	const AUTO(it, g_modules.find(path));
	if(it == g_modules.end()){
		return VAL_INIT;
	}
	return it->second.module;
}
boost::shared_ptr<Module> ModuleManager::load(const std::string &path){
	AUTO(module, get(path));
	if(module){
		return module;
	}

	const AUTO(newModule, boost::make_shared<Module>(path));
	boost::shared_ptr<const void> context;
	newModule->init(context);
	module = newModule;
	{
		const boost::unique_lock<boost::shared_mutex> lock(g_mutex);
		AUTO_REF(item, g_modules[path]);
		item.module = newModule;
		item.context.swap(context);
	}
	return module;
}
boost::shared_ptr<Module> ModuleManager::loadNoThrow(const std::string &path){
	try {
		return load(path);
	} catch(std::exception &e){
		LOG_ERROR("std::exception thrown while loading module: ", path, ", what = ", e.what());
	} catch(...){
		LOG_ERROR("Unknown exception thrown while loading module: ", path);
	}
	return VAL_INIT;
}
bool ModuleManager::unload(const std::string &path){
	const boost::unique_lock<boost::shared_mutex> lock(g_mutex);
	const AUTO(it, g_modules.find(path));
	if(it == g_modules.end()){
		return false;
	}

	try {
		LOG_INFO("Destroying context of module: ", path);
		g_modules.erase(it);
		LOG_INFO("Done destroying context of module: ", path);
	} catch(std::exception &e){
		LOG_ERROR("std::exception thrown while unloading module: ", path, ", what = ", e.what());
	} catch(...){
		LOG_ERROR("Unknown exception thrown while unloading module: ", path);
	}
	return true;
}

std::vector<ModuleInfo> ModuleManager::snapshot(){
	std::vector<ModuleInfo> ret;
	const boost::shared_lock<boost::shared_mutex> lock(g_mutex);
	for(AUTO(it, g_modules.begin()); it != g_modules.end(); ++it){
		ret.push_back(ModuleInfo());
		ModuleInfo &mi = ret.back();
		mi.path = it->first;
		mi.refCount = it->second.module.use_count();
	}
	return ret;
}
