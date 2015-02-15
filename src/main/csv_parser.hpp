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
	CsvParser()
		: m_row(0)
	{
	}
	explicit CsvParser(const char *file)
		: m_row(0)
	{
		load(file);
	}

public:
	void load(const char *file);
	bool loadNoThrow(const char *file);

	bool empty() const {
		return m_data.empty();
	}
	void clear(){
		m_data.clear();
		m_row = 0;
	}

	std::size_t rows() const {
		return m_data.size();
	}
	std::size_t tell() const {
		return m_row;
	}
	std::size_t seek(std::size_t row){
		if(row > m_data.size()){
			throw std::out_of_range("Poseidon::CsvParser::seek");
		}
		const AUTO(old, m_row);
		m_row = row;
		return old;
	}

	const std::string &getRaw(const char *key) const {
		return m_data.at(m_row).get(key);
	}

	template<typename T>
	bool get(T &val, const char *key) const {
		const AUTO_REF(str, m_data.at(m_row).get(key));
		if(str.empty()){
			return false;
		}
		val = boost::lexical_cast<T>(str);
		return true;
	}
	template<typename T, typename DefaultT>
	bool get(T &val, const char *key, const DefaultT &defVal) const {
		const AUTO_REF(str, m_data.at(m_row).get(key));
		if(str.empty()){
			val = defVal;
			return false;
		}
		val = boost::lexical_cast<T>(str);
		return true;
	}
	template<typename T, typename DefaultT>
	T get(const char *key, const DefaultT &defVal) const {
		const AUTO_REF(str, m_data.at(m_row).get(key));
		if(str.empty()){
			return T(defVal);
		}
		return boost::lexical_cast<T>(str);
	}
	template<typename T>
	T get(const char *key) const {
		const AUTO_REF(str, m_data.at(m_row).get(key));
		if(str.empty()){
			return T();
		}
		return boost::lexical_cast<T>(str);
	}
};

}

#endif
