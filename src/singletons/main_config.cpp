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
		LOG_POSEIDON_ERROR("Could not resolve path: ", path);
		DEBUG_THROW(SystemException);
	}
	try {
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Setting working directory: ", real_path);
		if(::chdir(real_path) != 0){
			LOG_POSEIDON_ERROR("Could not set working directory: ", real_path);
			DEBUG_THROW(SystemException);
		}
		::free(real_path);
	} catch(...){
		::free(real_path);
		throw;
	}
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
