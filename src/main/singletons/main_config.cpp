// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "main_config.hpp"
#include "../config_file.hpp"
#include "../log.hpp"
#include "../exception.hpp"

namespace Poseidon {

namespace {
	ConfigFile g_config;

	std::string getRealPath(const char *path){
		std::string ret;
		char *realPath = NULLPTR;
		try {
			realPath = ::realpath(path, NULLPTR);
			if(!realPath){
				LOG_POSEIDON_ERROR("Could not resolve path: ", path);
				DEBUG_THROW(SystemError);
			}
			ret = realPath;
			::free(realPath);
		} catch(...){
			::free(realPath);
			throw;
		}
		return ret;
	}
}

void MainConfig::setRunPath(const char *path){
	const AUTO(realPath, getRealPath(path));
	LOG_POSEIDON_INFO("Setting working directory: ", realPath);
	if(::chdir(realPath.c_str()) != 0){
		DEBUG_THROW(SystemError);
	}
}

void MainConfig::reload(){
	ConfigFile("main.conf").swap(g_config);
}
const ConfigFile &MainConfig::getConfigFile(){
	return g_config;
}

}
