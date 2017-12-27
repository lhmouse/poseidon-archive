// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CONFIG_FILE_HPP_
#define POSEIDON_CONFIG_FILE_HPP_

#include "cxx_ver.hpp"
#include "optional_map.hpp"
#include <boost/lexical_cast.hpp>

namespace Poseidon {

class ConfigFile {
private:
	OptionalMap m_contents;

public:
	ConfigFile()
		: m_contents()
	{ }
	explicit ConfigFile(const std::string &path)
		: m_contents()
	{
		load(path);
	}

public:
	void load(const std::string &path);
	int load_nothrow(const std::string &path);
	void save(const std::string &path);

	bool empty() const {
		return m_contents.empty();
	}
	void clear(){
		m_contents.clear();
	}

	const std::string &get_raw(const char *key) const {
		return m_contents.get(key);
	}
	std::size_t get_all_raw(boost::container::vector<std::string> &vals, const char *key, bool including_empty = false) const {
		const AUTO(range, m_contents.range(key));
		vals.reserve(vals.size() + static_cast<std::size_t>(std::distance(range.first, range.second)));
		std::size_t ret = 0;
		for(AUTO(it, range.first); it != range.second; ++it){
			if(it->second.empty()){
				if(!including_empty){
					continue;
				}
#ifdef POSEIDON_CXX11
				vals.emplace_back();
#else
				vals.push_back(VAL_INIT);
#endif
			} else {
				vals.push_back(it->second);
			}
			++ret;
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
		T val = VAL_INIT;
		get<T>(val, key);
		return val;
	}
	template<typename T, typename DefaultT>
	T get(const char *key, const DefaultT &def_val) const {
		T val = VAL_INIT;
		get<T, DefaultT>(val, key, def_val);
		return val;
	}

	template<typename T>
	std::size_t get_all(boost::container::vector<T> &vals, const char *key, bool including_empty = false) const {
		const AUTO(range, m_contents.range(key));
		vals.reserve(vals.size() + static_cast<std::size_t>(std::distance(range.first, range.second)));
		std::size_t ret = 0;
		for(AUTO(it, range.first); it != range.second; ++it){
			if(it->second.empty()){
				if(!including_empty){
					continue;
				}
#ifdef POSEIDON_CXX11
				vals.emplace_back();
#else
				vals.push_back(VAL_INIT);
#endif
			} else {
				vals.push_back(boost::lexical_cast<T>(it->second));
			}
			++ret;
		}
		return ret;
	}
	template<typename T>
	boost::container::vector<T> get_all(const char *key, bool including_empty = false) const {
		boost::container::vector<T> vals;
		const AUTO(range, m_contents.range(key));
		vals.reserve(static_cast<std::size_t>(std::distance(range.first, range.second)));
		for(AUTO(it, range.first); it != range.second; ++it){
			if(it->second.empty()){
				if(!including_empty){
					continue;
				}
#ifdef POSEIDON_CXX11
				vals.emplace_back();
#else
				vals.push_back(VAL_INIT);
#endif
			} else {
				vals.push_back(boost::lexical_cast<T>(it->second));
			}
		}
		return vals;
	}

	void swap(ConfigFile &rhs) NOEXCEPT {
		using std::swap;
		swap(m_contents, rhs.m_contents);
	}
};

inline void swap(ConfigFile &lhs, ConfigFile &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

}

#endif
