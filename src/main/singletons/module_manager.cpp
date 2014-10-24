#include "../precompiled.hpp"
#include "module_manager.hpp"
#include <map>
#include <boost/thread/mutex.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/type_traits/decay.hpp>
#include <dlfcn.h>
#include "main_config.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../exception.hpp"
#include "../multi_index_map.hpp"
using namespace Poseidon;

namespace {

struct ModuleMapElement {
	SharedNtmbs path;
	boost::shared_ptr<struct Module> module;

	ModuleContexts contexts;
};

MULTI_INDEX_MAP(ModuleMap, ModuleMapElement,
	UNIQUE_MEMBER_INDEX(path),
	MULTI_MEMBER_INDEX(module)
);

boost::mutex g_dlMutex;

boost::shared_mutex g_mutex;
ModuleMap g_modules;

struct DynamicLibraryCloser {
	CONSTEXPR void *operator()() NOEXCEPT {
		return VAL_INIT;
	}
	void operator()(void *handle) NOEXCEPT {
		const boost::mutex::scoped_lock lock(g_dlMutex);
		if(::dlclose(handle) != 0){
			LOG_WARN("Error unloading dynamic library: ", ::dlerror());
		}
	}
};

}

class Poseidon::Module : boost::noncopyable
	, public boost::enable_shared_from_this<Module> {
private:
	const SharedNtmbs m_path;
	ScopedHandle<DynamicLibraryCloser> m_handle;

public:
	explicit Module(SharedNtmbs path)
		: m_path(STD_MOVE(path))
	{
		LOG_INFO("Loading module: ", m_path);

		const boost::mutex::scoped_lock lock(g_dlMutex);
		m_handle.reset(::dlopen(m_path.get(), RTLD_NOW));
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
	const SharedNtmbs &getPath() const {
		return m_path;
	}

	void init(ModuleContexts &contexts){
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
		(*initProc)(shared_from_this(), contexts);
		LOG_INFO("Done initializing module: ", m_path);
	}
};

void ModuleManager::start(){
	LOG_INFO("Loading init modules...");

	std::vector<std::string> initModules;
	MainConfig::getAll(initModules, "init_module");
	for(AUTO(it, initModules.begin()); it != initModules.end(); ++it){
		load(*it);
	}
}
void ModuleManager::stop(){
	LOG_INFO("Unloading all modules...");

	g_modules.clear();
}

boost::shared_ptr<Module> ModuleManager::get(const char *path){
	const boost::shared_lock<boost::shared_mutex> lock(g_mutex);
	const AUTO(it, g_modules.find<0>(SharedNtmbs::createNonOwning(path)));
	if(it == g_modules.end()){
		return VAL_INIT;
	}
	return it->module;
}
boost::shared_ptr<Module> ModuleManager::load(const char *path){
	AUTO(module, get(path));
	if(!module){
		ModuleMapElement element;
		element.path = SharedNtmbs::createOwning(path);
		element.module = boost::make_shared<Module>(element.path);
		element.module->init(element.contexts);
		module = element.module;
		{
			const boost::unique_lock<boost::shared_mutex> lock(g_mutex);
			g_modules.insert(STD_MOVE(element));
		}
	}
	return module;
}
boost::shared_ptr<Module> ModuleManager::loadNoThrow(const char *path){
	try {
		return load(path);
	} catch(std::exception &e){
		LOG_ERROR("std::exception thrown while loading module: ", path,
			", what = ", e.what());
	} catch(...){
		LOG_ERROR("Unknown exception thrown while loading module: ", path);
	}
	return VAL_INIT;
}
bool ModuleManager::unload(const boost::shared_ptr<Module> &module){
	ModuleContexts contexts;
	{
		const boost::unique_lock<boost::shared_mutex> lock(g_mutex);
		const AUTO(it, g_modules.find<1>(module));
		if(it == g_modules.end<1>()){
			return false;
		}
		contexts.swap(const_cast<ModuleContexts &>(it->contexts));
		g_modules.erase<1>(it);
	}
	try {
		LOG_INFO("Destroying context of module: ", module->getPath());
		contexts.clear();
		LOG_INFO("Done destroying context of module: ", module->getPath());
	} catch(std::exception &e){
		LOG_ERROR("std::exception thrown while unloading module: ", module->getPath(),
			", what = ", e.what());
	} catch(...){
		LOG_ERROR("Unknown exception thrown while unloading module: ", module->getPath());
	}
	return true;
}

std::vector<ModuleSnapshotItem> ModuleManager::snapshot(){
	std::vector<ModuleSnapshotItem> ret;
	{
		const boost::shared_lock<boost::shared_mutex> lock(g_mutex);
		for(AUTO(it, g_modules.begin()); it != g_modules.end(); ++it){
			ret.push_back(ModuleSnapshotItem());
			ModuleSnapshotItem &mi = ret.back();
			mi.path = it->path;
			mi.refCount = it->module.use_count();
		}
	}
	return ret;
}
