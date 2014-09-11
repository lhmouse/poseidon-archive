#ifndef POSEIDON_SINGLETONS_CONFIG_FILE_HPP_
#define POSEIDON_SINGLETONS_CONFIG_FILE_HPP_

#include <string>
#include <boost/lexical_cast.hpp>

namespace Poseidon {

struct ConfigFile {
	static void reload(const char *path);

	static const std::string &get(const char *key);

	template<typename T>
	static T get(const char *key, T defaultVal){
		const std::string &str = get(key);
		if(str.empty()){
			return defaultVal;
		}
		return boost::lexical_cast<T>(str);
	}

private:
	ConfigFile();
};

}

#endif
