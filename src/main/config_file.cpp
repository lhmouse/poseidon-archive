#include "precompiled.hpp"
#include "config_file.hpp"
#include <fstream>
#include "log.hpp"
#include "exception.hpp"
using namespace Poseidon;

ConfigFile::ConfigFile(){
}
ConfigFile::ConfigFile(const char *path){
	if(!load(path)){
		DEBUG_THROW(Exception, "Failed to load config file");
	}
}
ConfigFile::ConfigFile(const SharedNtmbs &path){
	if(!load(path)){
		DEBUG_THROW(Exception, "Failed to load config file");
	}
}
ConfigFile::ConfigFile(const std::string &path){
	if(!load(path)){
		DEBUG_THROW(Exception, "Failed to load config file");
	}
}

bool ConfigFile::load(const char *path){
	LOG_INFO("Loading config file: ", path);

	std::ifstream ifs(path);
	if(!ifs){
		LOG_ERROR("Could not open config file: ", path);
		return false;
	}
	OptionalMap contents;
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
			LOG_ERROR("Error in config file on line ", count, ": '=' expected.");
			return false;
		}

		std::string key = line.substr(pos, equ);
		pos = key.find_last_not_of(" \t");
		if(pos == std::string::npos){
			LOG_ERROR("Error in config file on line ", count, ": Name expected.");
			return false;
		}
		key.erase(key.begin() + pos + 1, key.end());

		pos = line.find_first_not_of(" \t", equ + 1);
		if(pos != std::string::npos){
			line.erase(line.begin(), line.begin() + pos);
			pos = line.find_last_not_of(" \t");
			line.erase(line.begin() + pos + 1, line.end());

			LOG_DEBUG("Config: ", key, " = ", line);
			contents.append(STD_MOVE(key), STD_MOVE(line));
		}
	}
	m_contents.swap(contents);
	return true;
}
bool ConfigFile::load(const SharedNtmbs &path){
	return load(path.get());
}
bool ConfigFile::load(const std::string &path){
	return load(path.c_str());
}
