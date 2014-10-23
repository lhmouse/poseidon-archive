#ifndef POSEIDON_SINGLETONS_MAIN_CONFIG_HPP_
#define POSEIDON_SINGLETONS_MAIN_CONFIG_HPP_

#include "../config_file.hpp"

namespace Poseidon {

struct MainConfig {
	static void setRunPath(const char *path);
	static void reload();

	template<typename T>
	static bool get(T &val, const char *key){
		return getConfigFile().get<T>(val, key);
	}
	template<typename T>
	static std::size_t getAll(std::vector<T> &vals, const char *key){
		return getConfigFile().getAll<T>(vals, key);
	}

private:
	static const ConfigFile &getConfigFile();

	MainConfig();
};

}

#endif
