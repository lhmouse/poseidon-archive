#include "../precompiled.hpp"
#include "module.hpp"
#include <dlfcn.h>
#include "raii.hpp"
#include "log.hpp"
#include "exception.hpp"
using namespace Poseidon;

namespace {

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

public:
	explicit RealModule(std::string path)
		: Module(STD_MOVE(path))
	{
		m_handle.reset(::dlopen(getPath().c_str(), RTLD_NOW));
		if(!m_handle){
			const char *const error = ::dlerror();
			LOG_ERROR("Error loading dynamic library ", getPath(), ", error = ", error);
			DEBUG_THROW(Exception, error);
		}
	}
	~RealModule(){
		LOG_DEBUG("Uninitializing module: ", getPath());
	}

public:
	void init(){
		LOG_DEBUG("Initializing module: ", getPath());

		const AUTO(initProc, reinterpret_cast<
			void (*)(const boost::weak_ptr<const Module> &module)
			>(::dlsym(m_handle.get(), "poseidonModuleInit")));
		if(!initProc){
			const char *const error = ::dlerror();
			LOG_ERROR("Error getting address of poseidonModuleInit() in module ",
				getPath(), ", error = ", error);
			DEBUG_THROW(Exception, error);
		}
		(*initProc)(boost::weak_ptr<Module>(shared_from_this()));
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
