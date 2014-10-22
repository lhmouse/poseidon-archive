#ifndef POSEIDON_SINGLETONS_CONFIG_FILE_HPP_
#define POSEIDON_SINGLETONS_CONFIG_FILE_HPP_

#include <vector>
#include <string>
#include <boost/lexical_cast.hpp>

namespace Poseidon {

struct ConfigFile {
	static void setRunPath(const char *path);
	static void reload(const char *path);

	static const std::string &get(const char *key);
	static std::vector<std::string> getAll(const char *key);

	template<typename T>
	static bool get(T &val, const char *key){
		const std::string &str = get(key);
		if(str.empty()){
			return false;
		}
		val = boost::lexical_cast<T>(str);
		return true;
	}
	template<typename T>
	static std::size_t getAll(std::vector<T> &vals, const char *key){
		std::vector<std::string> tmp = getAll(key);
		const std::size_t ret = tmp.size();
		vals.reserve(vals.size() + ret);
		for(std::size_t i = 0; i < ret; ++i){
			vals.push_back(boost::lexical_cast<T>(tmp[i]));
		}
		return ret;
	}

private:
	ConfigFile();
};

}

#endif
