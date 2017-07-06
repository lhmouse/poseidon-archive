// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "main_config.hpp"
#include "../config_file.hpp"
#include "../log.hpp"
#include "../system_exception.hpp"
#include <limits.h>
#include <stdlib.h>

namespace Poseidon {

namespace {
	ConfigFile g_config;
}

void MainConfig::set_run_path(const char *path){
	char *const real_path = ::realpath(path, NULLPTR);
	if(!real_path){
		const int err_code = errno;
		LOG_POSEIDON_ERROR("Could not resolve path (errno was ", err_code, "): ", path);
		DEBUG_THROW(SystemException, err_code);
	}
	if(::chdir(real_path) != 0){
		const int err_code = errno;
		::free(real_path);
		LOG_POSEIDON_ERROR("Could not set working directory (errno was ", err_code, "): ", real_path);
		DEBUG_THROW(SystemException, err_code);
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Set new working directory: ", real_path);
	::free(real_path);
}
void MainConfig::reload(){
	static CONSTEXPR const char MAIN_CONF[] = "main.conf";
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Loading main config file: ", MAIN_CONF);
	ConfigFile config(MAIN_CONF);
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Done loading main config file: ", MAIN_CONF);
	g_config.swap(config);
}

const ConfigFile &MainConfig::get_config(){
	return g_config;
}

}
