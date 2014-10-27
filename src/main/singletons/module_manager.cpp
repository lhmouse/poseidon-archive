#include "../precompiled.hpp"
#include "module_manager.hpp"
#include <map>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/type_traits/decay.hpp>
#include <dlfcn.h>
#include "main_config.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../exception.hpp"
#include "../multi_index_map.hpp"
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
			LOG_WARN("Error unloading dynamic library: ", ::dlerror());
		}
	}
};

}

class Poseidon::Module : boost::noncopyable {
private:
	const ScopedHandle<DynamicLibraryCloser> m_handle;
	const SharedNtmbs m_realPath;
	void *const m_baseAddr;

public:
	Module(Move<ScopedHandle<DynamicLibraryCloser> > handle, SharedNtmbs realPath, void *baseAddr)
		: m_handle(STD_MOVE(handle)), m_realPath(STD_MOVE(realPath)), m_baseAddr(baseAddr)
	{
		LOG_INFO("Constructor of module: ", m_realPath);

		LOG_DEBUG("Handle: ", m_handle);
		LOG_DEBUG("Real path: ", m_realPath);
		LOG_DEBUG("Base addr: ", m_baseAddr);
	}
	~Module(){
		LOG_INFO("Destructor of module: ", m_realPath);

		LOG_DEBUG("Handle: ", m_handle);
		LOG_DEBUG("Real path: ", m_realPath);
		LOG_DEBUG("Base addr: ", m_baseAddr);
	}

public:
	void *handle() const {
		return m_handle.get();
	}
	const SharedNtmbs &realPath() const {
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
	const SharedNtmbs realPath;
	void *const baseAddr;

	mutable ModuleContexts contexts;

	explicit ModuleMapElement(boost::shared_ptr<Module> module_)
		: module(STD_MOVE(module_))
		, handle(module->handle()), realPath(module->realPath()), baseAddr(module->baseAddr())
	{
	}

#ifndef POSEIDON_CXX11
	// C++03 不提供转移构造函数，但是我们在这里不使用它，不需要定义。
	ModuleMapElement(Move<ModuleMapElement> rhs);
#endif
};

MULTI_INDEX_MAP(ModuleMap, ModuleMapElement,
	UNIQUE_MEMBER_INDEX(module),
	UNIQUE_MEMBER_INDEX(handle),
	MULTI_MEMBER_INDEX(realPath),
	UNIQUE_MEMBER_INDEX(baseAddr)
);

enum {
	IDX_MODULE,
	IDX_HANDLE,
	IDX_REAL_PATH,
	IDX_BASE_ADDR,
};

ModuleMap g_modules;

}

void ModuleManager::start(){
	std::vector<std::string> initModules;
	MainConfig::getAll(initModules, "init_module");
	for(AUTO(it, initModules.begin()); it != initModules.end(); ++it){
		LOG_INFO("Loading init module: ", *it);
		load(*it);
	}
}
void ModuleManager::stop(){
	LOG_INFO("Unloading all modules...");

	g_modules.clear();
}

boost::shared_ptr<Module> ModuleManager::load(const SharedNtmbs &path){
	const boost::recursive_mutex::scoped_lock lock(g_mutex);

	LOG_INFO("Checking whether module has already been loaded: ", path);
	ScopedHandle<DynamicLibraryCloser> handle(::dlopen(path.get(), RTLD_NOW | RTLD_NOLOAD));
	if(handle){
		LOG_DEBUG("Module already loaded, trying retrieving a shared_ptr from static map...");
		const AUTO(it, g_modules.find<IDX_HANDLE>(handle.get()));
		if(it != g_modules.end<IDX_HANDLE>()){
			LOG_DEBUG("Got shared_ptr from loaded module: ", it->realPath);
			return it->module;
		}
		LOG_DEBUG("Not found. Let's load as a new module.");
	}

	LOG_INFO("Loading new module: ", path);
	handle.reset(::dlopen(path.get(), RTLD_NOW | RTLD_DEEPBIND));
	if(!handle){
		const char *const error = ::dlerror();
		LOG_ERROR("Error loading dynamic library: ", error);
		DEBUG_THROW(Exception, error);
	}
	void *const initSym = ::dlsym(handle.get(), "poseidonModuleInit");
	if(!initSym){
		const char *const error = ::dlerror();
		LOG_ERROR("Error locating poseidonModuleInit(): ", error);
		DEBUG_THROW(Exception, error);
	}

	::Dl_info info;
	if(::dladdr(initSym, &info) == 0){
		const char *const error = ::dlerror();
		LOG_ERROR("Error getting real path: ", error);
		DEBUG_THROW(Exception, error);
	}
	SharedNtmbs realPath(info.dli_fname, true);
	void *const baseAddr = info.dli_fbase;

	AUTO(module, boost::make_shared<Module>(STD_MOVE(handle), realPath, baseAddr));

	LOG_INFO("Initializing module: ", realPath);
	ModuleContexts contexts;
	(*reinterpret_cast<VALUE_TYPE(::poseidonModuleInit)>(initSym))(module, contexts);
	LOG_INFO("Done initializing module: ", realPath);

	const AUTO(result, g_modules.insert(ModuleMapElement(module)));
	if(!result.second){
		LOG_ERROR("Duplicate module: module = ", module, ", handle = ", module->handle(),
			", real path = ", module->realPath(), ", base address = ", module->baseAddr());
		DEBUG_THROW(Exception, "Duplicate module");
	}
	result.first->contexts.swap(contexts);
	return module;
}
boost::shared_ptr<Module> ModuleManager::loadNoThrow(const SharedNtmbs &path){
	try {
		return load(path);
	} catch(...){
		return VAL_INIT;
	}
}
bool ModuleManager::unload(const boost::shared_ptr<Module> &module){
	const boost::recursive_mutex::scoped_lock lock(g_mutex);
	return g_modules.erase<IDX_MODULE>(module) > 0;
}
bool ModuleManager::unload(const SharedNtmbs &realPath){
	const boost::recursive_mutex::scoped_lock lock(g_mutex);
	return g_modules.erase<IDX_REAL_PATH>(realPath) > 0;
}
bool ModuleManager::unload(void *baseAddr){
	const boost::recursive_mutex::scoped_lock lock(g_mutex);
	return g_modules.erase<IDX_BASE_ADDR>(baseAddr) > 0;
}

boost::shared_ptr<Module> ModuleManager::assertCurrent(){
	const boost::recursive_mutex::scoped_lock lock(g_mutex);

	::Dl_info info;
	if(::dladdr(__builtin_return_address(0), &info) == 0){
		const char *const error = ::dlerror();
		LOG_ERROR("Error getting base address: ", error);
		DEBUG_THROW(Exception, error);
	}
	LOG_DEBUG("Current module = ", info.dli_fname, ", base address = ", info.dli_fbase);

	const AUTO(it, g_modules.find<IDX_BASE_ADDR>(info.dli_fbase));
	if(it == g_modules.end<IDX_BASE_ADDR>()){
		LOG_ERROR("Module was not loaded via ModuleManager: ", info.dli_fname);
		DEBUG_THROW(Exception, "Module was not loaded via ModuleManager");
	}
	return it->module;
}
std::vector<ModuleSnapshotItem> ModuleManager::snapshot(){
	std::vector<ModuleSnapshotItem> ret;
	{
		const boost::recursive_mutex::scoped_lock lock(g_mutex);
		for(AUTO(it, g_modules.begin()); it != g_modules.end(); ++it){
			ret.push_back(VAL_INIT);
			ModuleSnapshotItem &mi = ret.back();

			mi.realPath = it->realPath;
			mi.baseAddr = it->baseAddr;
			mi.refCount = it->module.use_count();
		}
	}
	return ret;
}
