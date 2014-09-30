#include "../precompiled.hpp"
#include "module.hpp"
#include <dlfcn.h>
#include "raii.hpp"
#include "log.hpp"
#include "exception.hpp"
using namespace Poseidon;

namespace {

typedef void (*ModuleInitProc)(const boost::weak_ptr<const Module> &module);
typedef void (*ModuleUninitProc)();

struct DynamicLibraryCloser {
	CONSTEXPR void *operator()() NOEXCEPT {
		return NULLPTR;
	}
	void operator()(void *handle) NOEXCEPT {
		if(::dlclose(handle) != 0){
			LOG_WARNING("Error unloading module: ", ::dlerror());
		}
	}
};

class RealModule : public Module {
private:
	ScopedHandle<DynamicLibraryCloser> m_handle;
	ModuleInitProc m_initProc;
	ModuleUninitProc m_uninitProc;

public:
	explicit RealModule(std::string path)
		: Module(STD_MOVE(path))
	{
		m_handle.reset(::dlopen(getPath().c_str(), RTLD_LAZY));
		if(!m_handle){
			const char *const error = ::dlerror();
			LOG_ERROR("Error loading dynamic library ", path, ", error = ", error);
			DEBUG_THROW(Exception, error);
		}
		m_initProc = reinterpret_cast<ModuleInitProc>(
			::dlsym(m_handle.get(), "poseidonModuleInit"));
		if(!m_initProc){
			const char *const error = ::dlerror();
			LOG_ERROR("Error getting address of poseidonModuleInit(), error = ", error);
			DEBUG_THROW(Exception, error);
		}
		m_uninitProc = reinterpret_cast<ModuleUninitProc>(
			::dlsym(m_handle.get(), "poseidonModuleUninit"));
		if(!m_uninitProc){
			const char *const error = ::dlerror();
			LOG_ERROR("Error getting address of poseidonModuleUninit(), error = ", error);
			DEBUG_THROW(Exception, error);
		}
	}
	~RealModule(){
		LOG_DEBUG("Uninitializing module: ", getPath());

		(*m_uninitProc)();
	}

public:
	void init(){
		LOG_DEBUG("Initializing module: ", getPath());

		(*m_initProc)(virtualWeakFromThis<RealModule>());
	}
};

}

boost::shared_ptr<Module> Module::load(std::string path){
	AUTO(module, boost::make_shared<RealModule>(STD_MOVE(path)));
	module->init();
	return module;
}

Module::Module(std::string path)
	: m_path(STD_MOVE(path))
{
}
Module::~Module(){
}
