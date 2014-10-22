#include "../../precompiled.hpp"
#include "config_file.hpp"
#include <cstring>
#include <fstream>
#include <map>
#include <stdlib.h>
#include <unistd.h>
#include "../log.hpp"
#include "../exception.hpp"
#include "../optional_map.hpp"
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

void ConfigFile::reload(const char *path){
	const AUTO(realPath, getRealPath(path));
	LOG_INFO("Loading config file: ", realPath);

	OptionalMap config;
	std::ifstream ifs(realPath.c_str());
	if(!ifs){
		LOG_FATAL("Could not open config file.");
		DEBUG_THROW(SystemError, EACCES);
	}
	std::string line;
	std::size_t count = 0;
	while(std::getline(ifs, line)){
		++count;
		std::size_t pos = line.find('#');
		if(pos != std::string::npos){
			line.erase(line.begin() + pos, line.end());
		}
		pos = line.find_first_not_of(" \t");
		if(pos == std::string::npos){
			continue;
		}
		std::size_t equ = line.find('=', pos);
		if(equ == std::string::npos){
			LOG_FATAL("Error in config file on line ", count, ": '=' expected.");
			DEBUG_THROW(SystemError, EINVAL);
		}

		std::string key = line.substr(pos, equ);
		pos = key.find_last_not_of(" \t");
		if(pos == std::string::npos){
			LOG_FATAL("Error in config file on line ", count, ": Name expected.");
			DEBUG_THROW(SystemError, EINVAL);
		}
		key.erase(key.begin() + pos + 1, key.end());

		pos = line.find_first_not_of(" \t", equ + 1);
		if(pos == std::string::npos){
			line.clear();
		} else {
			line.erase(line.begin(), line.begin() + pos);
			pos = line.find_last_not_of(" \t");
			line.erase(line.begin() + pos + 1, line.end());
		}

		LOG_DEBUG("Config: ", key, " = ", line);
		config.set(STD_MOVE(key), STD_MOVE(line));
	}
	config.swap(g_config);
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
