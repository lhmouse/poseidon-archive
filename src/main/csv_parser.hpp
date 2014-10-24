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

public:
	CsvParser(){
	}
	explicit CsvParser(const char *file){
		load(file);
	}
	explicit CsvParser(const std::string &file){
		load(file);
	}

public:
	void load(const char *file);
	void load(const std::string &file){
		load(file.c_str());
	}

	bool loadNoThrow(const char *file);
	bool loadNoThrow(const std::string &file){
		return loadNoThrow(file.c_str());
	}

	bool empty() const {
		return m_data.size();
	}
	void clear(){
		m_data.clear();
	}

	std::size_t rows() const {
		return m_data.size();
	}

	const std::string &get(std::size_t row, const SharedNtmbs &key){
		return m_data.at(row).get(key);
	}
	const std::string &get(std::size_t row, const char *key){
		return m_data.at(row).get(key);
	}
	const std::string &get(std::size_t row, const std::string &key){
		return m_data.at(row).get(key);
	}

	template<typename T>
	T get(std::size_t row, const SharedNtmbs &key){
		return boost::lexical_cast<T>(get(row, key));
	}
	template<typename T>
	T get(std::size_t row, const char *key){
		return boost::lexical_cast<T>(get(row, key));
	}
	template<typename T>
	T get(std::size_t row, const std::string &key){
		return boost::lexical_cast<T>(get(row, key));
	}
};

}

#endif
