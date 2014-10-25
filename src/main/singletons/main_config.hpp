#ifndef POSEIDON_SINGLETONS_MAIN_CONFIG_HPP_
#define POSEIDON_SINGLETONS_MAIN_CONFIG_HPP_

#include "../config_file.hpp"

namespace Poseidon {

struct MainConfig {
	static void setRunPath(const SharedNtmbs &path);
	static void reload();

	template<typename T>
	static bool get(T &val, const SharedNtmbs &key){
		return getConfigFile().get<T>(val, key);
	}
	template<typename T, typename DefaultT>
	static bool get(T &val, const SharedNtmbs &key, const DefaultT &defVal){
		return getConfigFile().get<T, DefaultT>(val, key, defVal);
	}
	template<typename T>
	static std::size_t getAll(std::vector<T> &vals, const SharedNtmbs &key,
		bool includingEmpty = false, bool truncates = false)
	{
		return getConfigFile().getAll<T>(vals, key, includingEmpty, truncates);
	}

private:
	static const ConfigFile &getConfigFile();

	MainConfig();
};

}

#endif
