// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "module_depository.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/type_traits/decay.hpp>
#include <dlfcn.h>
#include "main_config.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../exception.hpp"
#include "../multi_index_map.hpp"
#include "../module_raii.hpp"
using namespace Poseidon;

namespace {

// 注意 dl 系列的函数都不是线程安全的。
boost::recursive_mutex g_mutex;

struct DynamicLibraryCloser {
	CONSTEXPR void *operator()() NOEXCEPT {
		return VAL_INIT;
	}
	void operator()(void *handle) NOEXCEPT {
		const boost::recursive_mutex::scoped_lock lock(g_mutex);
		if(::dlclose(handle) != 0){
			LOG_POSEIDON_WARN("Error unloading dynamic library: ", ::dlerror());
		}
	}
};

}

class Poseidon::Module : NONCOPYABLE {
private:
	const UniqueHandle<DynamicLibraryCloser> m_handle;
	const SharedNts m_realPath;
	void *const m_baseAddr;

public:
	Module(UniqueHandle<DynamicLibraryCloser> handle, SharedNts realPath, void *baseAddr)
		: m_handle(STD_MOVE(handle)), m_realPath(STD_MOVE(realPath)), m_baseAddr(baseAddr)
	{
		LOG_POSEIDON_INFO("Constructor of module: ", m_realPath);

		LOG_POSEIDON_DEBUG("Handle: ", m_handle);
		LOG_POSEIDON_DEBUG("Real path: ", m_realPath);
		LOG_POSEIDON_DEBUG("Base addr: ", m_baseAddr);
	}
	~Module(){
		LOG_POSEIDON_INFO("Destructor of module: ", m_realPath);

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
	const boost::shared_ptr<Module> module;
	void *const handle;
	const SharedNts realPath;
	void *const baseAddr;

	mutable std::vector<boost::shared_ptr<void> > handles;

	explicit ModuleMapElement(boost::shared_ptr<Module> module_)
		: module(STD_MOVE(module_)), handle(module->handle())
		, realPath(module->realPath()), baseAddr(module->baseAddr())
	{
	}

#ifndef POSEIDON_CXX11
	// C++03 不提供转移构造函数，但是我们在这里不使用它，不需要定义。
	ModuleMapElement(Move<ModuleMapElement> rhs);
#endif
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

struct ModuleRaiiMapElement {
	void *const baseAddr;
	ModuleRaiiBase *const raii;

	ModuleRaiiMapElement(void *baseAddr_, ModuleRaiiBase *raii_)
		: baseAddr(baseAddr_), raii(raii_)
	{
	}

#ifndef POSEIDON_CXX11
	// C++03 不提供转移构造函数，但是我们在这里不使用它，不需要定义。
	ModuleRaiiMapElement(Move<ModuleRaiiMapElement> rhs);
#endif
};

MULTI_INDEX_MAP(ModuleRaiiMap, ModuleRaiiMapElement,
	MULTI_MEMBER_INDEX(baseAddr)
	UNIQUE_MEMBER_INDEX(raii)
)

enum {
	MRIDX_BASE_ADDR,
	MRIDX_RAII,
};

ModuleMap g_modules;
ModuleRaiiMap g_moduleRaiis;

}

void ModuleDepository::start(){
}
void ModuleDepository::stop(){
	LOG_POSEIDON_INFO("Unloading all modules...");

	std::vector<boost::weak_ptr<Module> > modules;
	{
		const boost::recursive_mutex::scoped_lock lock(g_mutex);
		modules.reserve(g_modules.size());
		for(AUTO(it, g_modules.begin()); it != g_modules.end(); ++it){
			modules.push_back(it->module);
		}
		g_modules.clear();
	}
	while(!modules.empty()){
		const AUTO(module, modules.back().lock());
		if(!module){
			modules.pop_back();
			continue;
		}
		LOG_POSEIDON_INFO("Waiting for module to unload: ", module->realPath());
		::sleep(1);
	}
}

boost::shared_ptr<Module> ModuleDepository::load(const char *path){
	const boost::recursive_mutex::scoped_lock lock(g_mutex);

	LOG_POSEIDON_INFO("Checking whether module has already been loaded: ", path);
	UniqueHandle<DynamicLibraryCloser> handle(::dlopen(path, RTLD_NOW | RTLD_NOLOAD));
	if(handle){
		LOG_POSEIDON_DEBUG("Module already loaded, trying retrieving a shared_ptr from static map...");
		const AUTO(it, g_modules.find<MIDX_HANDLE>(handle.get()));
		if(it != g_modules.end<MIDX_HANDLE>()){
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

	const AUTO(raiiRange, g_moduleRaiis.equalRange<MRIDX_BASE_ADDR>(baseAddr));
	std::vector<boost::shared_ptr<void> > handles;
	if(raiiRange.first == raiiRange.second){
		LOG_POSEIDON_INFO("No initialization is required: ", realPath);
	} else {
		LOG_POSEIDON_INFO("Initializing module: ", realPath);
		for(AUTO(it, raiiRange.first); it != raiiRange.second; ++it){
			boost::shared_ptr<void> handle(it->raii->init());
			if(!handle){
				continue;
			}
			handles.push_back(VAL_INIT);
			handles.back().swap(handle);
		}
		LOG_POSEIDON_INFO("Done initializing module: ", realPath);
	}

	const AUTO(result, g_modules.insert(ModuleMapElement(module)));
	if(!result.second){
		LOG_POSEIDON_ERROR("Duplicate module: module = ", module, ", handle = ", module->handle(),
			", real path = ", realPath, ", base address = ", baseAddr);
		DEBUG_THROW(Exception, SharedNts::observe("Duplicate module"));
	}
	result.first->handles.swap(handles);

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
	const boost::recursive_mutex::scoped_lock lock(g_mutex);
	return g_modules.erase<MIDX_MODULE>(module) > 0;
}
bool ModuleDepository::unload(void *baseAddr){
	const boost::recursive_mutex::scoped_lock lock(g_mutex);
	return g_modules.erase<MIDX_BASE_ADDR>(baseAddr) > 0;
}

std::vector<ModuleSnapshotItem> ModuleDepository::snapshot(){
	std::vector<ModuleSnapshotItem> ret;
	{
		const boost::recursive_mutex::scoped_lock lock(g_mutex);
		for(AUTO(it, g_modules.begin()); it != g_modules.end(); ++it){
			ret.push_back(ModuleSnapshotItem());
			ModuleSnapshotItem &mi = ret.back();

			mi.realPath = it->realPath;
			mi.baseAddr = it->baseAddr;
			mi.refCount = it->module.use_count();
		}
	}
	return ret;
}

void ModuleDepository::registerModuleRaii(ModuleRaiiBase *raii){
	const boost::recursive_mutex::scoped_lock lock(g_mutex);
	::Dl_info info;
	if(::dladdr(raii, &info) == 0){
		SharedNts error(::dlerror());
		LOG_POSEIDON_ERROR("Error getting base address: ", error);
		DEBUG_THROW(Exception, STD_MOVE(error));
	}
	if(!g_moduleRaiis.insert(ModuleRaiiMapElement(info.dli_fbase, raii)).second){
		LOG_POSEIDON_ERROR("Duplicate ModuleRaii? fbase = ", info.dli_fbase, ", raii = ", raii);
		DEBUG_THROW(Exception, SharedNts::observe("Duplicate ModuleRaii"));
	}
}
void ModuleDepository::unregisterModuleRaii(ModuleRaiiBase *raii){
	const boost::recursive_mutex::scoped_lock lock(g_mutex);
	g_moduleRaiis.erase<MRIDX_RAII>(raii);
}
