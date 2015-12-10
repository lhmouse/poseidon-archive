// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CSV_PARSER_HPP_
#define POSEIDON_CSV_PARSER_HPP_

#include "cxx_ver.hpp"
#include <vector>
#include <cstddef>
#include <boost/lexical_cast.hpp>
#include "optional_map.hpp"

namespace Poseidon {

class CsvParser {
private:
	std::vector<OptionalMap> m_data;
	std::size_t m_row;

public:
	CsvParser();
	explicit CsvParser(const char *file);

public:
	void load(const char *file);
	bool load_nothrow(const char *file);

	const std::vector<OptionalMap> &get_raw_data() const {
		return m_data;
	}
	std::vector<OptionalMap> &get_raw_data(){
		return m_data;
	}
	void set_raw_data(std::vector<OptionalMap> data){
		m_data.swap(data);
	}

	bool empty() const {
		return m_data.empty();
	}
	void clear();

	std::size_t rows() const {
		return m_data.size();
	}
	std::size_t tell() const {
		return m_row;
	}
	std::size_t seek(std::size_t row);

	bool fetch_row(){
		const AUTO(new_row, m_row + 1);
		if(new_row >= m_data.size()){
			return false;
		}
		m_row = new_row;
		return true;
	}

	const std::string &get_raw(const char *key) const;

	template<typename T>
	bool get(T &val, const char *key) const {
		const AUTO_REF(str, get_raw(key));
		if(str.empty()){
			return false;
		}
		val = boost::lexical_cast<T>(str);
		return true;
	}
	template<typename T, typename DefaultT>
	bool get(T &val, const char *key, const DefaultT &def_val) const {
		const AUTO_REF(str, get_raw(key));
		if(str.empty()){
			val = def_val;
			return false;
		}
		val = boost::lexical_cast<T>(str);
		return true;
	}
	template<typename T, typename DefaultT>
	T get(const char *key, const DefaultT &def_val) const {
		const AUTO_REF(str, get_raw(key));
		if(str.empty()){
			return T(def_val);
		}
		return boost::lexical_cast<T>(str);
	}
	template<typename T>
	T get(const char *key) const {
		const AUTO_REF(str, get_raw(key));
		if(str.empty()){
			return T();
		}
		return boost::lexical_cast<T>(str);
	}
};

}

#endif
