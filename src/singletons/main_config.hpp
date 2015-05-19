// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MAIN_CONFIG_HPP_
#define POSEIDON_SINGLETONS_MAIN_CONFIG_HPP_

#include "../config_file.hpp"

namespace Poseidon {

struct MainConfig {
	static void setRunPath(const char *path);

	static void reload();
	static const ConfigFile &getConfig();

	template<typename T>
	static bool get(T &val, const char *key){
		return getConfig().get<T>(val, key);
	}
	template<typename T, typename DefaultT>
	static bool get(T &val, const char *key, const DefaultT &defVal){
		return getConfig().get<T, DefaultT>(val, key, defVal);
	}
	template<typename T>
	static T get(const char *key){
		return getConfig().get<T>(key);
	}
	template<typename T, typename DefaultT>
	static T get(const char *key, const DefaultT &defVal){
		return getConfig().get<T, DefaultT>(key, defVal);
	}

	template<typename T>
	static std::size_t getAll(std::vector<T> &vals, const char *key, bool includingEmpty = false){
		return getConfig().getAll<T>(vals, key, includingEmpty);
	}
	template<typename T>
	static std::vector<T> getAll(const char *key, bool includingEmpty = false){
		return getConfig().getAll<T>(key, includingEmpty);
	}

private:
	MainConfig();
};

}

#endif
