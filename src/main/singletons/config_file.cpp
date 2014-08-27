#include "../../precompiled.hpp"
#include "config_file.hpp"
#include <cstring>
#include <fstream>
#include <map>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "../log.hpp"
#include "../exception.hpp"
using namespace Poseidon;

namespace {

struct KeyComparator {
	bool operator()(const boost::shared_ptr<const char> &lhs,
		const boost::shared_ptr<const char> &rhs) const
	{
		return std::strcmp(lhs.get(), rhs.get()) < 0;
	}
};

typedef std::map<
	boost::shared_ptr<const char>, std::string,
	KeyComparator
> ConfigMap;

const std::string EMPTY_STRING;

ConfigMap g_config;

}

void ConfigFile::reload(const char *path){
	LOG_INFO("Loading config from ", path, "...");

	ConfigMap config;
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

		AUTO(key, boost::make_shared<std::string>(line.substr(pos, equ)));
		key->resize(key->find_last_not_of(" \t") + 1);

		pos = line.find_first_not_of(" \t", equ + 1);
		std::string val = line.substr(pos);
		pos = val.find_last_not_of(" \t");
		if(pos == std::string::npos){
			val.clear();
		} else {
			val.resize(pos + 1);
		}

		LOG_DEBUG("Config: ", *key, " = ", val);
		config[boost::shared_ptr<const char>(key, key->c_str())].swap(val);
	}
	config.swap(g_config);
}
const std::string &ConfigFile::get(const char *key){
	AUTO(it, g_config.find(boost::shared_ptr<const char>(boost::shared_ptr<void>(), key)));
	if(it == g_config.end()){
		return EMPTY_STRING;
	}
	return it->second;
}
const std::string &ConfigFile::get(const std::string &key){
	return get(key.c_str());
}
