// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CONFIG_FILE_HPP_
#define POSEIDON_CONFIG_FILE_HPP_

#include "cxx_ver.hpp"
#include "optional_map.hpp"
#include <vector>
#include <algorithm>
#include <boost/lexical_cast.hpp>

namespace Poseidon {

class ConfigFile {
private:
	OptionalMap m_contents;

public:
	ConfigFile();
	explicit ConfigFile(const char *path);

public:
	void load(const char *path);
	int load_no_throw(const char *path);

	bool empty() const {
		return m_contents.empty();
	}
	void clear(){
		m_contents.clear();
	}

	void swap(ConfigFile &rhs) NOEXCEPT {
		m_contents.swap(rhs.m_contents);
	}

	const std::string &get_raw(const char *key) const {
		return m_contents.get(key);
	}
	std::size_t get_all_raw(std::vector<std::string> &vals, const char *key, bool including_empty = false) const {
		const AUTO(range, m_contents.range(key));
		vals.reserve(static_cast<std::size_t>(std::distance(range.first, range.second)));
		std::size_t ret = 0;
		for(AUTO(it, range.first); it != range.second; ++it){
			if(including_empty || !it->second.empty()){
				vals.push_back(it->second);
				++ret;
			}
		}
		return ret;
	}

	template<typename T>
	bool get(T &val, const char *key) const {
		const AUTO_REF(str, m_contents.get(key));
		if(str.empty()){
			return false;
		}
		val = boost::lexical_cast<T>(str);
		return true;
	}
	template<typename T, typename DefaultT>
	bool get(T &val, const char *key, const DefaultT &def_val) const {
		const AUTO_REF(str, m_contents.get(key));
		if(str.empty()){
			val = static_cast<T>(def_val);
			return false;
		}
		val = boost::lexical_cast<T>(str);
		return true;
	}
	template<typename T>
	T get(const char *key) const {
		T val;
		get(val, key, T());
		return val;
	}
	template<typename T, typename DefaultT>
	T get(const char *key, const DefaultT &def_val) const {
		T val;
		get(val, key, def_val);
		return val;
	}

	template<typename T>
	std::size_t get_all(std::vector<T> &vals, const char *key, bool including_empty = false) const {
		const AUTO(range, m_contents.range(key));
		vals.reserve(static_cast<std::size_t>(std::distance(range.first, range.second)));
		std::size_t ret = 0;
		for(AUTO(it, range.first); it != range.second; ++it){
			if(including_empty || !it->second.empty()){
				vals.push_back(boost::lexical_cast<T>(it->second));
				++ret;
			}
		}
		return ret;
	}
	template<typename T>
	std::vector<T> get_all(const char *key, bool including_empty = false) const {
		std::vector<T> vals;
		get_all(vals, key, including_empty);
		return vals;
	}
};

inline void swap(ConfigFile &lhs, ConfigFile &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

}

#endif
