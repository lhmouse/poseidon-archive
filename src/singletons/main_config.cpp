// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "main_config.hpp"
#include "../config_file.hpp"
#include "../log.hpp"
#include "../system_exception.hpp"

namespace Poseidon {

namespace {
	ConfigFile g_config;

	std::string get_real_path(const char *path){
		std::string ret;
		char *real_path = NULLPTR;
		try {
			real_path = ::realpath(path, NULLPTR);
			if(!real_path){
				LOG_POSEIDON_ERROR("Could not resolve path: ", path);
				DEBUG_THROW(SystemException);
			}
			ret = real_path;
			::free(real_path);
		} catch(...){
			::free(real_path);
			throw;
		}
		return ret;
	}
}

void MainConfig::set_run_path(const char *path){
	const AUTO(real_path, get_real_path(path));
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Setting working directory: real_path = ", real_path);
	if(::chdir(real_path.c_str()) != 0){
		DEBUG_THROW(SystemException);
	}
}

void MainConfig::reload(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Loading main.conf...");
	ConfigFile("main.conf").swap(g_config);
}
const ConfigFile &MainConfig::get_config(){
	return g_config;
}

}
