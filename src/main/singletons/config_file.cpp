#include "../../precompiled.hpp"
#include "config_file.hpp"
#include "../config_file.hpp"
#include "../log.hpp"
#include "../exception.hpp"
using namespace Poseidon;

namespace {

OptionalMap g_config;

std::string getRealPath(const char *path){
	std::string ret;
	char *realPath = VAL_INIT;
	try {
		realPath = ::realpath(path, VAL_INIT);
		if(!realPath){
			LOG_ERROR("Could not resolve path: ", path);
			DEBUG_THROW(SystemError, errno);
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

void ConfigFile::setRunPath(const char *path){
	const AUTO(realPath, getRealPath(path));
	LOG_INFO("Setting working directory: ", realPath);

	if(::chdir(realPath.c_str()) != 0){
		DEBUG_THROW(SystemError, errno);
	}
}

void ConfigFile::reload(){
	const AUTO(realPath, getRealPath("main.conf"));
	LOG_INFO("Loading config file: ", realPath);

	OptionalMap config;
	if(!loadConfigFile(config, realPath)){
		LOG_ERROR("Error loading main config file: ", realPath);
		DEBUG_THROW(Exception, "Error loading main config file");
	}
	g_config.swap(config);
}

const std::string &ConfigFile::get(const char *key){
	return g_config.get(key);
}
std::vector<std::string> ConfigFile::getAll(const char *key){
	std::vector<std::string> ret;
	const AUTO(range, g_config.range(key));
	ret.reserve(std::distance(range.first, range.second));
	for(AUTO(it, range.first); it != range.second; ++it){
		ret.push_back(it->second);
	}
	return ret;
}
