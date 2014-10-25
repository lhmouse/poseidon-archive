#include "../precompiled.hpp"
#include "main_config.hpp"
#include "../config_file.hpp"
#include "../log.hpp"
#include "../exception.hpp"
using namespace Poseidon;

namespace {

ConfigFile g_config;

std::string getRealPath(const char *path){
	std::string ret;
	char *realPath = VAL_INIT;
	try {
		realPath = ::realpath(path, VAL_INIT);
		if(!realPath){
			LOG_ERROR("Could not resolve path: ", path);
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

void MainConfig::setRunPath(const SharedNtmbs &path){
	const AUTO(realPath, getRealPath(path.get()));
	LOG_INFO("Setting working directory: ", realPath);
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
