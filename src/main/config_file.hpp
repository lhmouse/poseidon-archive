#ifndef POSEIDON_CONFIG_FILE_HPP_
#define POSEIDON_CONFIG_FILE_HPP_

#include "../cxx_ver.hpp"
#include "optional_map.hpp"
#include <algorithm>
#include <boost/lexical_cast.hpp>

namespace Poseidon {

class ConfigFile {
private:
	OptionalMap m_contents;

public:
	ConfigFile();
	explicit ConfigFile(const char *path);
	explicit ConfigFile(const SharedNtmbs &path);
	explicit ConfigFile(const std::string &path);

public:
	bool load(const char *path);
	bool load(const SharedNtmbs &path);
	bool load(const std::string &path);

	bool empty() const {
		return m_contents.empty();
	}
	void clear(){
		m_contents.clear();
	}

	void swap(ConfigFile &rhs) NOEXCEPT {
		m_contents.swap(rhs.m_contents);
	}

	template<typename T>
	bool get(T &val, const char *key) const {
		const std::string &str = m_contents.get(key);
		if(str.empty()){
			return false;
		}
		val = boost::lexical_cast<T>(str);
		return true;
	}
	template<typename T>
	bool get(T &val, const SharedNtmbs &key) const {
		return get<T>(val, key.get());
	}
	template<typename T>
	bool get(T &val, const std::string &key) const {
		return get<T>(val, key.c_str());
	}

	template<typename T>
	std::size_t getAll(std::vector<T> &vals, const char *key) const {
		std::pair<OptionalMap::const_iterator,
			OptionalMap::const_iterator> range = m_contents.range(key);
		std::size_t ret = 0;
		while(range.first != range.second){
			vals.push_back(boost::lexical_cast<T>(range.first->second));
			++range.first;
			++ret;
		}
		return ret;
	}
	template<typename T>
	std::size_t getAll(std::vector<T> &vals, const SharedNtmbs &key) const {
		return getAll<T>(vals, key.get());
	}
	template<typename T>
	std::size_t getAll(std::vector<T> &vals, const std::string &key) const {
		return getAll<T>(vals, key.c_str());
	}
};

static inline void swap(ConfigFile &lhs, ConfigFile &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

}

#endif
