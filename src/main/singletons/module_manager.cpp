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
	const boost::shared_ptr<Module> module;
	const SharedNtmbs realPath;
	void *const baseAddr;

	mutable ModuleContexts contexts;

	ModuleMapElement(boost::shared_ptr<Module> module_,
		SharedNtmbs realPath_, void *baseAddr_)
		: module(STD_MOVE(module_))
		, realPath(STD_MOVE(realPath_)), baseAddr(baseAddr_)
	{
	}
};

MULTI_INDEX_MAP(ModuleMap, ModuleMapElement,
	UNIQUE_MEMBER_INDEX(module),
	MULTI_MEMBER_INDEX(realPath),
	MULTI_MEMBER_INDEX(baseAddr)
);

enum {
	IDX_MODULE,
	IDX_REAL_PATH,
	IDX_BASE_ADDR,
};

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
	ScopedHandle<DynamicLibraryCloser> m_handle;
	SharedNtmbs m_realPath;
	void *m_baseAddr;

public:
	~Module(){
		if(m_handle){
			LOG_INFO("Unloading module: ", m_realPath);

			LOG_DEBUG("Handle: ", m_handle);
			LOG_DEBUG("Real path: ", m_realPath);
			LOG_DEBUG("Base addr: ", m_baseAddr);
		}
	}

public:
	const SharedNtmbs &realPath() const {
		return m_realPath;
	}
	void *baseAddr() const {
		return m_baseAddr;
	}

	void load(ModuleContexts &contexts, const SharedNtmbs &path){
		assert(!m_handle);

		LOG_INFO("Loading module: ", path);

		VALUE_TYPE(::poseidonModuleInit) initProc;
		{
			const boost::mutex::scoped_lock lock(g_dlMutex);

			m_handle.reset(::dlopen(path.get(), RTLD_NOW));
			if(!m_handle){
				const char *const error = ::dlerror();
				LOG_ERROR("Error loading dynamic library: ", error);
				DEBUG_THROW(Exception, error);
			}

			void *const initSym = ::dlsym(m_handle.get(), "poseidonModuleInit");
			if(!initSym){
				const char *const error = ::dlerror();
				LOG_ERROR("Error getting address of poseidonModuleInit(): ", error);
				DEBUG_THROW(Exception, error);
			}
			initProc = reinterpret_cast<VALUE_TYPE(::poseidonModuleInit)>(initSym);

			::Dl_info info;
			if(::dladdr(initSym, &info) == 0){
				const char *const error = ::dlerror();
				LOG_ERROR("Error getting real path: ", error);
				DEBUG_THROW(Exception, error);
			}
			SharedNtmbs(info.dli_fname, true).swap(m_realPath);
			m_baseAddr = info.dli_fbase;
		}

		LOG_DEBUG("Handle: ", m_handle);
		LOG_DEBUG("Real path: ", m_realPath);
		LOG_DEBUG("Base addr: ", m_baseAddr);

		LOG_INFO("Initializing module: ", m_realPath);
		(*initProc)(shared_from_this(), contexts);
		LOG_INFO("Done initializing module: ", m_realPath);
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

	ModuleMap modules;
	{
		const boost::unique_lock<boost::shared_mutex> lock(g_mutex);
		modules.swap(g_modules);
	}
}

boost::shared_ptr<Module> ModuleManager::get(const SharedNtmbs &realPath){
	const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
	const AUTO(it, g_modules.find<IDX_REAL_PATH>(realPath));
	if(it == g_modules.end<IDX_REAL_PATH>()){
		return VAL_INIT;
	}
	return it->module;
}
boost::shared_ptr<Module> ModuleManager::assertCurrent(){
	const char *realPath;
	void *baseAddr;
	{
		const boost::mutex::scoped_lock lock(g_dlMutex);
		::Dl_info info;
		if(::dladdr(__builtin_return_address(0), &info) == 0){
			const char *const error = ::dlerror();
			LOG_ERROR("Error getting base addr: ", error);
			DEBUG_THROW(Exception, error);
		}
		realPath = info.dli_fname;
		baseAddr = info.dli_fbase;
	}
	LOG_DEBUG("Base addr: ", baseAddr);

	const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
	const AUTO(it, g_modules.find<IDX_BASE_ADDR>(baseAddr));
	if(it == g_modules.end<IDX_BASE_ADDR>()){
		LOG_ERROR("Module was not loaded via ModuleManager: ", realPath);
		DEBUG_THROW(Exception, "Module was not loaded via ModuleManager");
	}
	return it->module;
}
boost::shared_ptr<Module> ModuleManager::load(const SharedNtmbs &path){
	AUTO(module, boost::make_shared<Module>());
	ModuleContexts contexts;
	module->load(contexts, path);
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		const AUTO(result, g_modules.insert(
			ModuleMapElement(module, module->realPath(), module->baseAddr())));
		if(result.second){
			contexts.swap(result.first->contexts);
		}
	}
	return module;
}
boost::shared_ptr<Module> ModuleManager::loadNoThrow(const SharedNtmbs &path){
	try {
		return load(path);
	} catch(Exception &){
		return VAL_INIT;
	}
}
bool ModuleManager::unload(const boost::shared_ptr<Module> &module){
	SharedNtmbs realPath;
	ModuleContexts contexts;
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		const AUTO(it, g_modules.find<IDX_MODULE>(module));
		if(it == g_modules.end<IDX_MODULE>()){
			return false;
		}
		realPath = it->realPath;
		contexts.swap(it->contexts);
		g_modules.erase<IDX_MODULE>(it);
	}
	try {
		LOG_INFO("Destroying context of module: ", realPath);
		contexts.clear();
		LOG_INFO("Done destroying context of module: ", realPath);
	} catch(std::exception &e){
		LOG_ERROR("std::exception thrown while unloading module: ", realPath);
	} catch(...){
		LOG_ERROR("Unknown exception thrown while unloading module: ", realPath);
	}
	return true;
}
bool ModuleManager::unload(const SharedNtmbs &path){
	SharedNtmbs realPath;
	ModuleContexts contexts;
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		const AUTO(it, g_modules.find<IDX_REAL_PATH>(path));
		if(it == g_modules.end<IDX_REAL_PATH>()){
			return false;
		}
		realPath = it->realPath;
		contexts.swap(it->contexts);
		g_modules.erase<IDX_REAL_PATH>(it);
	}
	try {
		LOG_INFO("Destroying context of module: ", realPath);
		contexts.clear();
		LOG_INFO("Done destroying context of module: ", realPath);
	} catch(std::exception &e){
		LOG_ERROR("std::exception thrown while unloading module: ", realPath);
	} catch(...){
		LOG_ERROR("Unknown exception thrown while unloading module: ", realPath);
	}
	return true;
}

ModuleSnapshotItem ModuleManager::snapshot(const boost::shared_ptr<Module> &module){
	ModuleSnapshotItem ret;
	ret.realPath = module->realPath();
	ret.refCount = module.use_count();
	return ret;
}
std::vector<ModuleSnapshotItem> ModuleManager::snapshot(){
	std::vector<ModuleSnapshotItem> ret;
	{
		const boost::shared_lock<boost::shared_mutex> lock(g_mutex);
		for(AUTO(it, g_modules.begin()); it != g_modules.end(); ++it){
			ret.push_back(ModuleSnapshotItem());
			ModuleSnapshotItem &mi = ret.back();
			mi.realPath = it->module->realPath();
			mi.refCount = it->module.use_count();
		}
	}
	return ret;
}
