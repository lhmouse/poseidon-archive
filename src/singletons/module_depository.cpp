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
}

namespace {
	class Module : NONCOPYABLE {
	private:
		UniqueHandle<DynamicLibraryCloser> m_dl_handle;
		void *m_base_address;
		SharedNts m_real_path;

	public:
		Module(Move<UniqueHandle<DynamicLibraryCloser> > dl_handle, SharedNts real_path, void *base_address)
			: m_dl_handle(STD_MOVE(dl_handle)), m_base_address(base_address), m_real_path(STD_MOVE(real_path))
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
	};

	struct ModuleMapElement {
		boost::shared_ptr<Module> module;
		boost::shared_ptr<HandleStack> handles;

		void *dl_handle;
		void *base_address;

		ModuleMapElement(boost::shared_ptr<Module> module_, boost::shared_ptr<HandleStack> handles_)
			: module(STD_MOVE(module_)), handles(STD_MOVE(handles_))
			, dl_handle(module->get_dl_handle()), base_address(module->get_base_address())
		{ }
	};
	MULTI_INDEX_MAP(ModuleMap, ModuleMapElement,
		UNIQUE_MEMBER_INDEX(dl_handle)
		UNIQUE_MEMBER_INDEX(base_address)
	)
	ModuleMap g_module_map;

	struct ModuleRaiiMapElement {
		ModuleRaiiBase *raii;
		std::pair<void *, long> base_address_priority;

		ModuleRaiiMapElement(ModuleRaiiBase *raii_, void *base_address, long priority_)
			: raii(raii_), base_address_priority(base_address, priority_)
		{ }
	};
	MULTI_INDEX_MAP(ModuleRaiiMap, ModuleRaiiMapElement,
		UNIQUE_MEMBER_INDEX(raii)
		MULTI_MEMBER_INDEX(base_address_priority)
	)
	ModuleRaiiMap g_module_raii_map;
}

void ModuleDepository::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting module depository...");
}
void ModuleDepository::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unloading all modules...");

	ModuleMap module_map;
	{
		const RecursiveMutex::UniqueLock lock(g_mutex);
		module_map.swap(g_module_map);
	}
	for(;;){
		const AUTO(it, module_map.begin());
		if(it == module_map.end()){
			break;
		}
		LOG_POSEIDON_INFO("Unloading module: ", it->module->get_real_path());
		module_map.erase(it);
	}
}

void *ModuleDepository::load(const std::string &path){
	PROFILE_ME;

	void *base_address;
	const RecursiveMutex::UniqueLock lock(g_mutex);
	{
		LOG_POSEIDON_INFO("Loading module: ", path);
		UniqueHandle<DynamicLibraryCloser> handle;
		if(!handle.reset(::dlopen(path.c_str(), RTLD_NOW | RTLD_NODELETE | RTLD_DEEPBIND))){
			const char *const error = ::dlerror();
			LOG_POSEIDON_ERROR("Error loading dynamic library: ", error);
			DEBUG_THROW(Exception, SharedNts(error));
		}
		AUTO(mod_it, g_module_map.find<0>(handle.get()));
		if(mod_it == g_module_map.end()){
			void *const init_sym = ::dlsym(handle.get(), "_init");
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
			AUTO(module, boost::make_shared<Module>(STD_MOVE(handle), SharedNts(info.dli_fname), info.dli_fbase));

			LOG_POSEIDON_INFO("Initializing NEW module: ", module->get_real_path());
			AUTO(handles, boost::make_shared<HandleStack>());
			const AUTO(raii_range_lower, g_module_raii_map.lower_bound<1>(std::make_pair(info.dli_fbase, LONG_MIN)));
			const AUTO(raii_range_upper, g_module_raii_map.upper_bound<1>(std::make_pair(info.dli_fbase, LONG_MAX)));
			for(AUTO(it, raii_range_lower); it != raii_range_upper; ++it){
				LOG_POSEIDON_DEBUG("> Performing module RAII initialization: raii = ", static_cast<void *>(it->raii));
				it->raii->init(*handles);
			}
			LOG_POSEIDON_INFO("Done initializing module: ", module->get_real_path());

			const AUTO(result, g_module_map.insert(ModuleMapElement(STD_MOVE(module), STD_MOVE(handles))));
			DEBUG_THROW_ASSERT(result.second);
			mod_it = result.first;
		}
		base_address = mod_it->module->get_base_address();
	}
	return base_address;
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
bool ModuleDepository::unload(void *base_address){
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(g_mutex);
	return g_module_map.erase<1>(base_address) != 0;
}

std::vector<ModuleDepository::SnapshotElement> ModuleDepository::snapshot(){
	PROFILE_ME;

	std::vector<SnapshotElement> ret;
	{
		const RecursiveMutex::UniqueLock lock(g_mutex);
		ret.reserve(g_module_map.size());
		for(AUTO(it, g_module_map.begin()); it != g_module_map.end(); ++it){
			SnapshotElement elem = { };
			elem.dl_handle = it->module->get_dl_handle();
			elem.base_address = it->module->get_base_address();
			elem.real_path = it->module->get_real_path();
			ret.push_back(STD_MOVE(elem));
		}
	}
	return ret;
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
	const AUTO(result, g_module_raii_map.insert(ModuleRaiiMapElement(raii, info.dli_fbase, priority)));
	if(!result.second){
		LOG_POSEIDON_ERROR("Duplicate ModuleRaii? raii = ", static_cast<void *>(raii));
		DEBUG_THROW(Exception, sslit("Duplicate ModuleRaii"));
	}
}
void ModuleDepository::unregister_module_raii(ModuleRaiiBase *raii){
	PROFILE_ME;

	const RecursiveMutex::UniqueLock lock(g_mutex);
	g_module_raii_map.erase<0>(raii);
}

}
