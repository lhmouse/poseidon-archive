// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "module_depository.hpp"
#include <boost/type_traits/decay.hpp>
#include <dlfcn.h>
#include "main_config.hpp"
#include "../recursive_mutex.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../raii.hpp"
#include "../exception.hpp"
#include "../multi_index_map.hpp"
#include "../module_raii.hpp"

namespace Poseidon {

namespace {
	// 注意 dl 系列的函数都不是线程安全的。
	RecursiveMutex g_mutex;

	struct ModuleRaiiMapElement {
		ModuleRaiiBase *raii;
		std::pair<void *, long> base_address_priority;
	};
	MULTI_INDEX_MAP(ModuleRaiiMap, ModuleRaiiMapElement,
		UNIQUE_MEMBER_INDEX(raii)
		MULTI_MEMBER_INDEX(base_address_priority)
	)
	ModuleRaiiMap g_module_raii_map;

	struct DynamicLibraryCloser {
		CONSTEXPR void *operator()() NOEXCEPT {
			return NULLPTR;
		}
		void operator()(void *handle) NOEXCEPT {
			const RecursiveMutex::UniqueLock lock(g_mutex);
			if(::dlclose(handle) != 0){
				const char *const error = ::dlerror();
				LOG_POSEIDON_WARNING("Error unloading dynamic library: ", error);
			}
		}
	};

	class Module : NONCOPYABLE {
	private:
		UniqueHandle<DynamicLibraryCloser> m_dl_handle;
		void *m_base_address;
		SharedNts m_real_path;

		HandleStack m_handles;

	public:
		Module(Move<UniqueHandle<DynamicLibraryCloser> > dl_handle, void *base_address, SharedNts real_path,
			Move<HandleStack> handles)
			: m_dl_handle(STD_MOVE(dl_handle)), m_base_address(base_address), m_real_path(STD_MOVE(real_path))
			, m_handles(STD_MOVE(handles))
		{
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Constructor of module: ", m_real_path);
			LOG_POSEIDON_DEBUG("> dl_handle = ", m_dl_handle, ", base_address = ", m_base_address, ", real_path = ", m_real_path);
		}
		~Module(){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Destructor of module: ", m_real_path);
			LOG_POSEIDON_DEBUG("> dl_handle = ", m_dl_handle, ", base_address = ", m_base_address, ", real_path = ", m_real_path);
		}

	public:
		void *get_dl_handle() const {
			return m_dl_handle.get();
		}
		void *get_base_address() const {
			return m_base_address;
		}
		const SharedNts &get_real_path() const {
			return m_real_path;
		}

		const HandleStack &get_handle_stack() const {
			return m_handles;
		}
		HandleStack &get_handle_stack(){
			return m_handles;
		}
	};

	struct ModuleMapElement {
		boost::shared_ptr<Module> module;

		void *dl_handle;
		void *base_address;
	};
	MULTI_INDEX_MAP(ModuleMap, ModuleMapElement,
		UNIQUE_MEMBER_INDEX(dl_handle)
		UNIQUE_MEMBER_INDEX(base_address)
	)
	ModuleMap g_module_map;
}

void ModuleDepository::register_module_raii(ModuleRaiiBase *raii, long priority){
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(g_mutex);
	::Dl_info info;
	if(::dladdr(raii, &info) == 0){
		const char *const error = ::dlerror();
		LOG_POSEIDON_ERROR("Error getting base address: ", error);
		DEBUG_THROW(Exception, SharedNts(error));
	}
	ModuleRaiiMapElement elem = { raii, std::make_pair(info.dli_fbase, priority) };
	const AUTO(result, g_module_raii_map.insert(STD_MOVE(elem)));
	if(!result.second){
		LOG_POSEIDON_ERROR("Duplicate ModuleRaii? raii = ", static_cast<void *>(raii));
		DEBUG_THROW(Exception, sslit("Duplicate ModuleRaii"));
	}
}
void ModuleDepository::unregister_module_raii(ModuleRaiiBase *raii) NOEXCEPT {
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(g_mutex);
	const AUTO(it, g_module_raii_map.find<0>(raii));
	if(it == g_module_raii_map.end()){
		LOG_POSEIDON_ERROR("ModuleRaii not found? raii = ", static_cast<void *>(raii));
		return;
	}
	g_module_raii_map.erase<0>(it);
}

void ModuleDepository::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting module depository...");
}
void ModuleDepository::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unloading all modules...");

	for(;;){
		const AUTO(it, g_module_map.begin());
		if(it == g_module_map.end()){
			break;
		}
		LOG_POSEIDON_INFO("Unloading module: ", it->module->get_real_path());
		g_module_map.erase(it);
	}
}

void *ModuleDepository::load(const std::string &path){
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(g_mutex);
	LOG_POSEIDON_INFO("Loading module: ", path);
	UniqueHandle<DynamicLibraryCloser> dl_handle;
	if(!dl_handle.reset(::dlopen(path.c_str(), RTLD_NOW | RTLD_NODELETE | RTLD_DEEPBIND))){
		const char *const error = ::dlerror();
		LOG_POSEIDON_ERROR("Error loading dynamic library: ", error);
		DEBUG_THROW(Exception, SharedNts(error));
	}
	AUTO(it, g_module_map.find<0>(dl_handle.get()));
	if(it != g_module_map.end()){
		LOG_POSEIDON_WARNING("Module already loaded: ", path);
	} else {
		void *const init_sym = ::dlsym(dl_handle.get(), "_init");
		if(!init_sym){
			const char *const error = ::dlerror();
			LOG_POSEIDON_ERROR("Error locating `_init()`: ", error);
			DEBUG_THROW(Exception, SharedNts(error));
		}
		::Dl_info info;
		if(::dladdr(init_sym, &info) == 0){
			const char *const error = ::dlerror();
			LOG_POSEIDON_ERROR("Error getting real path and base address: ", error);
			DEBUG_THROW(Exception, SharedNts(error));
		}
		HandleStack handles;
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Initializing NEW module: ", info.dli_fname);
		const AUTO(raii_range_lower, g_module_raii_map.lower_bound<1>(std::make_pair(info.dli_fbase, LONG_MIN)));
		const AUTO(raii_range_upper, g_module_raii_map.upper_bound<1>(std::make_pair(info.dli_fbase, LONG_MAX)));
		for(AUTO(raii_it, raii_range_lower); raii_it != raii_range_upper; ++raii_it){
			LOG_POSEIDON_DEBUG("> Performing module RAII initialization: raii = ", static_cast<void *>(raii_it->raii));
			raii_it->raii->init(handles);
		}
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Done initializing module: ", info.dli_fname);
		const AUTO(module, boost::make_shared<Module>(STD_MOVE(dl_handle), info.dli_fbase, SharedNts(info.dli_fname), STD_MOVE(handles)));
		ModuleMapElement elem = { module, module->get_dl_handle(), module->get_base_address() };
		const AUTO(result, g_module_map.insert(STD_MOVE(elem)));
		DEBUG_THROW_ASSERT(result.second);
		it = result.first;
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Loaded module: base_address = ", module->get_base_address(), ", real_path = ", module->get_real_path());
	}
	return it->module->get_base_address();
}
void *ModuleDepository::load_nothrow(const std::string &path)
try {
	PROFILE_ME;

	return load(path);
} catch(Exception &e){
	LOG_POSEIDON_ERROR("Exception thrown while loading module: path = ", path, ", what = ", e.what());
	return NULLPTR;
} catch(std::exception &e){
	LOG_POSEIDON_ERROR("std::exception thrown while loading module: path = ", path, ", what = ", e.what());
	return NULLPTR;
} catch(...){
	LOG_POSEIDON_ERROR("Unknown exception thrown while loading module: path = ", path);
	return NULLPTR;
}
bool ModuleDepository::unload(void *base_address) NOEXCEPT {
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(g_mutex);
	const AUTO(it, g_module_map.find<1>(base_address));
	if(it == g_module_map.end<1>()){
		LOG_POSEIDON_WARNING("Module not found: base_address = ", base_address);
		return false;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
		"Unloading module: base_address = ", base_address, ", real_path = ", it->module->get_real_path());
	g_module_map.erase<1>(it);
	return true;
}

void ModuleDepository::snapshot(std::vector<ModuleDepository::SnapshotElement> &ret){
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(g_mutex);
	ret.reserve(ret.size() + g_module_map.size());
	for(AUTO(it, g_module_map.begin()); it != g_module_map.end(); ++it){
		SnapshotElement elem = { };
		elem.dl_handle = it->module->get_dl_handle();
		elem.base_address = it->module->get_base_address();
		elem.real_path = it->module->get_real_path();
		ret.push_back(STD_MOVE(elem));
	}
}

}
