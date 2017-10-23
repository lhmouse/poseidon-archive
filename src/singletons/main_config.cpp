// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "main_config.hpp"
#include "../config_file.hpp"
#include "../log.hpp"
#include "../system_exception.hpp"
#include "../raii.hpp"
#include "../mutex.hpp"
#include <limits.h>
#include <stdlib.h>

namespace Poseidon {

namespace {
	struct RealPathDeleter {
		CONSTEXPR char *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(char *ptr) const NOEXCEPT {
			::free(ptr);
		}
	};

	Mutex g_mutex;
	boost::shared_ptr<ConfigFile> g_config;
}

void MainConfig::set_run_path(const char *path){
	UniqueHandle<RealPathDeleter> real_path;
	if(!real_path.reset(::realpath(path, NULLPTR))){
		const int err_code = errno;
		LOG_POSEIDON_ERROR("Could not resolve path (errno was ", err_code, "): ", path);
		DEBUG_THROW(SystemException, err_code);
	}
	if(::chdir(real_path.get()) != 0){
		const int err_code = errno;
		LOG_POSEIDON_ERROR("Could not set working directory (errno was ", err_code, "): ", real_path);
		DEBUG_THROW(SystemException, err_code);
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Set new working directory: ", real_path);
}
void MainConfig::reload(){
	static CONSTEXPR const char MAIN_CONF[] = "main.conf";
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Loading main config file: ", MAIN_CONF);
	AUTO(config, boost::make_shared<ConfigFile>(MAIN_CONF));
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Done loading main config file: ", MAIN_CONF);

	const Mutex::UniqueLock lock(g_mutex);
	g_config.swap(config);
}

boost::shared_ptr<const ConfigFile> MainConfig::get_config(){
	const Mutex::UniqueLock lock(g_mutex);
	if(!g_config){
		LOG_POSEIDON_ERROR("Main config file has not been loaded.");
		DEBUG_THROW(Exception, sslit("Main config file has not been loaded"));
	}
	return g_config;
}

}
