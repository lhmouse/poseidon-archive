// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MAIN_CONFIG_HPP_
#define POSEIDON_SINGLETONS_MAIN_CONFIG_HPP_

#include "../config_file.hpp"
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class MainConfig {
private:
	MainConfig();

public:
	static void set_run_path(const char *path);
	static void reload();

	static boost::shared_ptr<const ConfigFile> get_file();

	static bool get_raw(std::string &val, const char *key){
		return get_file()->get_raw(val, key);
	}
	static const std::string &get_raw(const char *key){
		return get_file()->get_raw(key);
	}

	static std::size_t get_all_raw(boost::container::vector<std::string> &vals, const char *key, bool including_empty = false){
		return get_file()->get_all_raw(vals, key, including_empty);
	}
	static boost::container::vector<std::string> get_all_raw(const char *key, bool including_empty = false){
		return get_file()->get_all_raw(key, including_empty);
	}

	template<typename T>
	static bool get(T &val, const char *key){
		return get_file()->get<T>(val, key);
	}
	template<typename T>
	static T get(const char *key){
		return get_file()->get<T>(key);
	}

	template<typename T, typename DefaultT>
	static bool get(T &val, const char *key, const DefaultT &def_val){
		return get_file()->get<T>(val, key, def_val);
	}
	template<typename T, typename DefaultT>
	static T get(const char *key, const DefaultT &def_val){
		return get_file()->get<T>(key, def_val);
	}

	template<typename T>
	static std::size_t get_all(boost::container::vector<T> &vals, const char *key, bool including_empty = false){
		return get_file()->get_all<T>(vals, key, including_empty);
	}
	template<typename T>
	static boost::container::vector<T> get_all(const char *key, bool including_empty = false){
		return get_file()->get_all<T>(key, including_empty);
	}
};

}

#endif
