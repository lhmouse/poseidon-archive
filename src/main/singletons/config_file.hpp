#ifndef POSEIDON_SINGLETONS_CONFIG_FILE_HPP_
#define POSEIDON_SINGLETONS_CONFIG_FILE_HPP_

#include <vector>
#include <string>
#include <boost/lexical_cast.hpp>

namespace Poseidon {

struct ConfigFile {
	static void reload(const char *path);

	static const std::string &get(const char *key);
	static std::vector<std::string> getAll(const char *key);

	template<typename T>
	static T get(const char *key, T defaultVal){
		const std::string &str = get(key);
		if(str.empty()){
			return defaultVal;
		}
		return boost::lexical_cast<T>(str);
	}
	template<typename T>
	static std::vector<T> getAll(const char *key){
		std::vector<std::string> tmp = getAll(key);
		const std::size_t size = tmp.size();

		std::vector<T> ret;
		ret.reserve(size);
		for(std::size_t i = 0; i < size; ++i){
			ret.push_back(boost::lexical_cast<T>(tmp[i]));
		}
		return ret;
	}

private:
	ConfigFile();
};

}

#endif
