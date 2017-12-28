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
	CONSTEXPR const char g_main_conf_name[] = "main.conf";

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
	DEBUG_THROW_UNLESS(real_path.reset(::realpath(path, NULLPTR)), SystemException);
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Set new working directory: ", real_path);
	DEBUG_THROW_UNLESS(::chdir(real_path.get()) == 0, SystemException);
}
void MainConfig::reload(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Loading main config file: ", g_main_conf_name);
	AUTO(config, boost::make_shared<ConfigFile>(g_main_conf_name));
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Done loading main config file: ", g_main_conf_name);
	const Mutex::UniqueLock lock(g_mutex);
	g_config.swap(config);
}

boost::shared_ptr<const ConfigFile> MainConfig::get_file(){
	const Mutex::UniqueLock lock(g_mutex);
	DEBUG_THROW_UNLESS(g_config, Exception, sslit("Main config file has not been loaded"));
	return g_config;
}

}
