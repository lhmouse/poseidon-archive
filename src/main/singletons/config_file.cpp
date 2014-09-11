#include "../../precompiled.hpp"
#include "config_file.hpp"
#include <cstring>
#include <fstream>
#include <map>
#include <boost/shared_ptr.hpp>
#include "../log.hpp"
#include "../exception.hpp"
#include "../optional_map.hpp"
using namespace Poseidon;

namespace {

OptionalMap g_config;

}

void ConfigFile::reload(const char *path){
	LOG_INFO("Loading config from ", path, "...");

	OptionalMap config;
	std::ifstream ifs(path);
	if(!ifs.good()){
		LOG_FATAL("Could not open config file ", path);
		DEBUG_THROW(SystemError, ENOENT);
	}
	std::string line;
	std::size_t count = 0;
	while(std::getline(ifs, line)){
		++count;
		std::size_t pos = line.find('#');
		if(pos != std::string::npos){
			line.resize(pos);
		}
		pos = line.find_first_not_of(" \t");
		if(pos == std::string::npos){
			continue;
		}
		std::size_t equ = line.find('=', pos);
		if(equ == pos){
			LOG_FATAL("Error in config file on line ", count, ": Name expected.");
			DEBUG_THROW(SystemError, EINVAL);
		}
		if(equ == std::string::npos){
			LOG_FATAL("Error in config file on line ", count, ": '=' expected.");
			DEBUG_THROW(SystemError, EINVAL);
		}

		std::string key = line.substr(pos, equ);
		key.resize(key.find_last_not_of(" \t") + 1);

		pos = line.find_first_not_of(" \t", equ + 1);
		std::string val = line.substr(pos);
		pos = val.find_last_not_of(" \t");
		if(pos == std::string::npos){
			val.clear();
		} else {
			val.resize(pos + 1);
		}

		LOG_DEBUG("Config: ", key, " = ", val);
		config.set(key, val);
	}
	config.swap(g_config);
}
const std::string &ConfigFile::get(const char *key){
	return g_config.get(key);
}
