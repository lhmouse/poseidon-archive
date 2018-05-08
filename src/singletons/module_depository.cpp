// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

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
	Recursive_mutex g_mutex;

	struct Module_raii_map_element {
		// Indices.
		Module_raii_base *raii;
		std::pair<void *, long> base_address_priority;
	};
	MULTI_INDEX_MAP(Module_raii_map, Module_raii_map_element,
		UNIQUE_MEMBER_INDEX(raii)
		MULTI_MEMBER_INDEX(base_address_priority)
	);
	Module_raii_map g_module_raii_map;

	struct Dynamic_library_closer {
		CONSTEXPR void *operator()() NOEXCEPT {
			return NULLPTR;
		}
		void operator()(void *handle) NOEXCEPT {
			const Recursive_mutex::Unique_lock lock(g_mutex);
			if(::dlclose(handle) != 0){
				const char *const error = ::dlerror();
				LOG_POSEIDON_WARNING("Error unloading dynamic library: ", error);
			}
		}
	};

	class Module : NONCOPYABLE {
	private:
		Unique_handle<Dynamic_library_closer> m_dl_handle;
		void *m_base_address;
		Rcnts m_real_path;

		Handle_stack m_handles;

	public:
		Module(Move<Unique_handle<Dynamic_library_closer> > dl_handle, void *base_address, Rcnts real_path, Move<Handle_stack> handles)
			: m_dl_handle(STD_MOVE(dl_handle)), m_base_address(base_address), m_real_path(STD_MOVE(real_path))
			, m_handles(STD_MOVE(handles))
		{
			LOG_POSEIDON(Logger::special_major | Logger::level_info, "Constructor of module: ", m_real_path);
			LOG_POSEIDON_DEBUG("> dl_handle = ", m_dl_handle, ", base_address = ", m_base_address, ", real_path = ", m_real_path);
		}
		~Module(){
			LOG_POSEIDON(Logger::special_major | Logger::level_info, "Destructor of module: ", m_real_path);
			LOG_POSEIDON_DEBUG("> dl_handle = ", m_dl_handle, ", base_address = ", m_base_address, ", real_path = ", m_real_path);
		}

	public:
		void *get_dl_handle() const {
			return m_dl_handle.get();
		}
		void *get_base_address() const {
			return m_base_address;
		}
		const Rcnts &get_real_path() const {
			return m_real_path;
		}

		const Handle_stack &get_handle_stack() const {
			return m_handles;
		}
		Handle_stack &get_handle_stack(){
			return m_handles;
		}
	};

	struct Module_map_element {
		// Invariants.
		boost::shared_ptr<Module> module;
		// Indices.
		void *dl_handle;
		void *base_address;
	};
	MULTI_INDEX_MAP(Module_map, Module_map_element,
		UNIQUE_MEMBER_INDEX(dl_handle)
		UNIQUE_MEMBER_INDEX(base_address)
	);
	Module_map g_module_map;
}

void Module_depository::register_module_raii(Module_raii_base *raii, long priority){
	PROFILE_ME;

	const Recursive_mutex::Unique_lock lock(g_mutex);
	::Dl_info info;
	DEBUG_THROW_UNLESS(::dladdr(raii, &info), Exception, Rcnts::view("Error getting base address"));
	Module_raii_map_element elem = { raii, std::make_pair(info.dli_fbase, priority) };
	const AUTO(result, g_module_raii_map.insert(STD_MOVE(elem)));
	DEBUG_THROW_UNLESS(result.second, Exception, Rcnts::view("Duplicate Module_raii"));
}
void Module_depository::unregister_module_raii(Module_raii_base *raii) NOEXCEPT {
	PROFILE_ME;

	const Recursive_mutex::Unique_lock lock(g_mutex);
	const AUTO(it, g_module_raii_map.find<0>(raii));
	if(it == g_module_raii_map.end()){
		LOG_POSEIDON_ERROR("Module_raii not found? raii = ", static_cast<void *>(raii));
		return;
	}
	g_module_raii_map.erase<0>(it);
}

void Module_depository::start(){
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Starting module depository...");
}
void Module_depository::stop(){
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Unloading all modules...");

	const Recursive_mutex::Unique_lock lock(g_mutex);
	g_module_map.clear();
}

void *Module_depository::load(const std::string &path){
	PROFILE_ME;

	const Recursive_mutex::Unique_lock lock(g_mutex);
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Loading module: ", path);
	Unique_handle<Dynamic_library_closer> dl_handle;
	DEBUG_THROW_UNLESS(dl_handle.reset(::dlopen(path.c_str(), RTLD_NOW | RTLD_NODELETE | RTLD_DEEPBIND)), Exception, Rcnts(::dlerror()));
	AUTO(it, g_module_map.find<0>(dl_handle.get()));
	if(it != g_module_map.end()){
		LOG_POSEIDON_WARNING("Module already loaded: ", path);
	} else {
		void *const init_sym = ::dlsym(dl_handle.get(), "_init");
		DEBUG_THROW_UNLESS(init_sym, Exception, Rcnts(::dlerror()));
		::Dl_info info;
		DEBUG_THROW_UNLESS(::dladdr(init_sym, &info), Exception, Rcnts(::dlerror()));
		Handle_stack handles;
		LOG_POSEIDON(Logger::special_major | Logger::level_info, "Initializing NEW module: ", info.dli_fname);
		const AUTO(raii_range_lower, g_module_raii_map.lower_bound<1>(std::make_pair(info.dli_fbase, LONG_MIN)));
		const AUTO(raii_range_upper, g_module_raii_map.upper_bound<1>(std::make_pair(info.dli_fbase, LONG_MAX)));
		for(AUTO(raii_it, raii_range_lower); raii_it != raii_range_upper; ++raii_it){
			LOG_POSEIDON_DEBUG("> Performing module initialization: raii = ", static_cast<void *>(raii_it->raii));
			raii_it->raii->init(handles);
		}
		LOG_POSEIDON(Logger::special_major | Logger::level_info, "Done initializing module: ", info.dli_fname);
		const AUTO(module, boost::make_shared<Module>(STD_MOVE(dl_handle), info.dli_fbase, Rcnts(info.dli_fname), STD_MOVE(handles)));
		Module_map_element elem = { module, module->get_dl_handle(), module->get_base_address() };
		const AUTO(result, g_module_map.insert(STD_MOVE(elem)));
		DEBUG_THROW_ASSERT(result.second);
		it = result.first;
		LOG_POSEIDON(Logger::special_major | Logger::level_info, "Loaded module: base_address = ", module->get_base_address(), ", real_path = ", module->get_real_path());
	}
	return it->module->get_base_address();
}
void *Module_depository::load_nothrow(const std::string &path)
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
bool Module_depository::unload(void *base_address) NOEXCEPT {
	PROFILE_ME;

	const Recursive_mutex::Unique_lock lock(g_mutex);
	const AUTO(it, g_module_map.find<1>(base_address));
	if(it == g_module_map.end<1>()){
		LOG_POSEIDON_WARNING("Module not found: base_address = ", base_address);
		return false;
	}
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Unloading module: base_address = ", base_address, ", real_path = ", it->module->get_real_path());
	g_module_map.erase<1>(it);
	return true;
}

void Module_depository::snapshot(boost::container::vector<Module_depository::Snapshot_element> &ret){
	PROFILE_ME;

	const Recursive_mutex::Unique_lock lock(g_mutex);
	ret.reserve(ret.size() + g_module_map.size());
	for(AUTO(it, g_module_map.begin()); it != g_module_map.end(); ++it){
		Snapshot_element elem = { };
		elem.dl_handle = it->module->get_dl_handle();
		elem.base_address = it->module->get_base_address();
		elem.real_path = it->module->get_real_path();
		ret.push_back(STD_MOVE(elem));
	}
}

}
