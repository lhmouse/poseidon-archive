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
	explicit ConfigFile(const SharedNtmbs &path);

public:
	bool load(const SharedNtmbs &path);

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
	bool get(T &val, const SharedNtmbs &key) const {
		const std::string &str = m_contents.get(key);
		if(str.empty()){
			return false;
		}
		val = boost::lexical_cast<T>(str);
		return true;
	}
	template<typename T, typename DefaultT>
	bool get(T &val, const SharedNtmbs &key, const DefaultT &defVal) const {
		const std::string &str = m_contents.get(key);
		if(str.empty()){
			val = defVal;
			return false;
		}
		val = boost::lexical_cast<T>(str);
		return true;
	}

	template<typename T>
	std::size_t getAll(std::vector<T> &vals, const SharedNtmbs &key, bool truncates = false) const {
		if(truncates){
			vals.clear();
		}
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
};

static inline void swap(ConfigFile &lhs, ConfigFile &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

}

#endif
