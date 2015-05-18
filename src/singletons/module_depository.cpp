// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "module_depository.hpp"
#include <boost/type_traits/decay.hpp>
#include <dlfcn.h>
#include "main_config.hpp"
#include "../mutex.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../exception.hpp"
#include "../multi_index_map.hpp"
#include "../module_raii.hpp"

namespace Poseidon {

namespace {
	// 注意 dl 系列的函数都不是线程安全的。
	Mutex g_mutex(true);

	struct DynamicLibraryCloser {
		CONSTEXPR void *operator()() NOEXCEPT {
			return VAL_INIT;
		}
		void operator()(void *handle) NOEXCEPT {
			const Mutex::UniqueLock lock(g_mutex);
			if(::dlclose(handle) != 0){
				LOG_POSEIDON_WARNING("Error unloading dynamic library: ", ::dlerror());
			}
		}
	};
}

class Module : NONCOPYABLE {
private:
	const UniqueHandle<DynamicLibraryCloser> m_handle;
	const SharedNts m_realPath;
	void *const m_baseAddr;

public:
	Module(UniqueHandle<DynamicLibraryCloser> handle, SharedNts realPath, void *baseAddr)
		: m_handle(STD_MOVE(handle)), m_realPath(STD_MOVE(realPath)), m_baseAddr(baseAddr)
	{
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Constructor of module: ", m_realPath);

		LOG_POSEIDON_DEBUG("Handle: ", m_handle);
		LOG_POSEIDON_DEBUG("Real path: ", m_realPath);
		LOG_POSEIDON_DEBUG("Base addr: ", m_baseAddr);
	}
	~Module(){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Destructor of module: ", m_realPath);

		LOG_POSEIDON_DEBUG("Handle: ", m_handle);
		LOG_POSEIDON_DEBUG("Real path: ", m_realPath);
		LOG_POSEIDON_DEBUG("Base addr: ", m_baseAddr);
	}

public:
	void *handle() const {
		return m_handle.get();
	}
	const SharedNts &realPath() const {
		return m_realPath;
	}
	void *baseAddr() const {
		return m_baseAddr;
	}
};

namespace {

struct ModuleMapElement {
	boost::shared_ptr<Module> module;
	void *handle;
	SharedNts realPath;
	void *baseAddr;

	HandleStack handles;

	ModuleMapElement(boost::shared_ptr<Module> module_, HandleStack handles_)
		: module(STD_MOVE(module_)), handle(module->handle())
		, realPath(module->realPath()), baseAddr(module->baseAddr())
		, handles(STD_MOVE(handles_))
	{
	}
};

MULTI_INDEX_MAP(ModuleMap, ModuleMapElement,
	UNIQUE_MEMBER_INDEX(module)
	UNIQUE_MEMBER_INDEX(handle)
	MULTI_MEMBER_INDEX(realPath)
	UNIQUE_MEMBER_INDEX(baseAddr)
)

enum {
	MIDX_MODULE,
	MIDX_HANDLE,
	MIDX_REAL_PATH,
	MIDX_BASE_ADDR,
};

ModuleMap g_moduleMap;

struct ModuleRaiiMapElement {
	ModuleRaiiBase *raii;
	long priority;

	void *baseAddr;

	ModuleRaiiMapElement(ModuleRaiiBase *raii_, long priority_, void *baseAddr_)
		: raii(raii_), priority(priority_), baseAddr(baseAddr_)
	{
	}
};

MULTI_INDEX_MAP(ModuleRaiiMap, ModuleRaiiMapElement,
	UNIQUE_MEMBER_INDEX(raii)
	MULTI_MEMBER_INDEX(priority)
)

enum {
	MRIDX_RAII,
	MRIDX_PRIORITY,
};

ModuleRaiiMap g_moduleRaiiMap;

}

void ModuleDepository::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting module depository...");
}
void ModuleDepository::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unloading all modules...");

	std::vector<boost::weak_ptr<Module> > modules;
	{
		const Mutex::UniqueLock lock(g_mutex);
		modules.reserve(g_moduleMap.size());
		for(AUTO(it, g_moduleMap.begin()); it != g_moduleMap.end(); ++it){
			modules.push_back(it->module);
		}
		g_moduleMap.clear();
	}
	while(!modules.empty()){
		const AUTO(module, modules.back().lock());
		if(!module){
			modules.pop_back();
			continue;
		}
		LOG_POSEIDON_INFO("Waiting for module to unload: ", module->realPath());
		::usleep(100000);
	}
}

boost::shared_ptr<Module> ModuleDepository::load(const char *path){
	const Mutex::UniqueLock lock(g_mutex);

	LOG_POSEIDON_INFO("Checking whether module has already been loaded: ", path);
	UniqueHandle<DynamicLibraryCloser> handle(::dlopen(path, RTLD_NOW | RTLD_NOLOAD));
	if(handle){
		LOG_POSEIDON_DEBUG("Module already loaded, trying retrieving a shared_ptr from static map...");
		const AUTO(it, g_moduleMap.find<MIDX_HANDLE>(handle.get()));
		if(it != g_moduleMap.end<MIDX_HANDLE>()){
			LOG_POSEIDON_DEBUG("Got shared_ptr from loaded module: ", it->realPath);
			return it->module;
		}
		LOG_POSEIDON_DEBUG("Not found. Let's load as a new module.");
	}

	LOG_POSEIDON_INFO("Loading new module: ", path);
	if(!handle.reset(::dlopen(path, RTLD_NOW | RTLD_NODELETE | RTLD_DEEPBIND))){
		SharedNts error(::dlerror());
		LOG_POSEIDON_ERROR("Error loading dynamic library: ", error);
		DEBUG_THROW(Exception, STD_MOVE(error));
	}

	void *const initSym = ::dlsym(handle.get(), "_init");
	if(!initSym){
		SharedNts error(::dlerror());
		LOG_POSEIDON_ERROR("Error locating _init(): ", error);
		DEBUG_THROW(Exception, STD_MOVE(error));
	}
	::Dl_info info;
	if(::dladdr(initSym, &info) == 0){
		SharedNts error(::dlerror());
		LOG_POSEIDON_ERROR("Error getting real path and base address: ", error);
		DEBUG_THROW(Exception, STD_MOVE(error));
	}
	SharedNts realPath(info.dli_fname);
	void *const baseAddr = info.dli_fbase;

	AUTO(module, boost::make_shared<Module>(STD_MOVE(handle), realPath, baseAddr));

	HandleStack handles;
	LOG_POSEIDON_INFO("Initializing module: ", realPath);
	for(AUTO(it, g_moduleRaiiMap.begin<MRIDX_PRIORITY>()); it != g_moduleRaiiMap.end<MRIDX_PRIORITY>(); ++it){
		if(it->baseAddr != baseAddr){
			continue;
		}
		it->raii->init(handles);
	}
	LOG_POSEIDON_INFO("Done initializing module: ", realPath);

	const AUTO(result, g_moduleMap.insert(ModuleMapElement(module, STD_MOVE(handles))));
	if(!result.second){
		LOG_POSEIDON_ERROR("Duplicate module: module = ", static_cast<void *>(module.get()),
			", handle = ", module->handle(),", realPath = ", realPath, ", baseAddr = ", baseAddr);
		DEBUG_THROW(Exception, sslit("Duplicate module"));
	}

	return module;
}
boost::shared_ptr<Module> ModuleDepository::loadNoThrow(const char *path){
	try {
		return load(path);
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("std::exception thrown while loading module: path = ", path, ", what = ", e.what());
		return VAL_INIT;
	} catch(...){
		LOG_POSEIDON_ERROR("Unknown exception thrown while loading module: path = ", path);
		return VAL_INIT;
	}
}
bool ModuleDepository::unload(const boost::shared_ptr<Module> &module){
	const Mutex::UniqueLock lock(g_mutex);
	return g_moduleMap.erase<MIDX_MODULE>(module) > 0;
}
bool ModuleDepository::unload(void *baseAddr){
	const Mutex::UniqueLock lock(g_mutex);
	return g_moduleMap.erase<MIDX_BASE_ADDR>(baseAddr) > 0;
}

std::vector<ModuleDepository::SnapshotElement> ModuleDepository::snapshot(){
	std::vector<SnapshotElement> ret;
	{
		const Mutex::UniqueLock lock(g_mutex);
		for(AUTO(it, g_moduleMap.begin()); it != g_moduleMap.end(); ++it){
			ret.push_back(SnapshotElement());
			SnapshotElement &mi = ret.back();

			mi.realPath = it->realPath;
			mi.baseAddr = it->baseAddr;
			mi.refCount = static_cast<unsigned long>(it->module.use_count()); // fvck
		}
	}
	return ret;
}

void ModuleDepository::registerModuleRaii(ModuleRaiiBase *raii, long priority){
	const Mutex::UniqueLock lock(g_mutex);
	::Dl_info info;
	if(::dladdr(raii, &info) == 0){
		SharedNts error(::dlerror());
		LOG_POSEIDON_ERROR("Error getting base address: ", error);
		DEBUG_THROW(Exception, STD_MOVE(error));
	}
	if(!g_moduleRaiiMap.insert(ModuleRaiiMapElement(raii, priority, info.dli_fbase)).second){
		LOG_POSEIDON_ERROR("Duplicate ModuleRaii? fbase = ", info.dli_fbase, ", raii = ", static_cast<void *>(raii));
		DEBUG_THROW(Exception, sslit("Duplicate ModuleRaii"));
	}
}
void ModuleDepository::unregisterModuleRaii(ModuleRaiiBase *raii){
	const Mutex::UniqueLock lock(g_mutex);
	g_moduleRaiiMap.erase<MRIDX_RAII>(raii);
}

}
