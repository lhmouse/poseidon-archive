// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CSV_DOCUMENT_HPP_
#define POSEIDON_CSV_DOCUMENT_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <iosfwd>
#include <stdexcept>
#include <cstddef>
#include <boost/container/vector.hpp>
#include <boost/container/map.hpp>
#include "shared_nts.hpp"

namespace Poseidon {

extern const std::string &empty_string() NOEXCEPT;

class CsvDocument {
private:
	boost::container::map<SharedNts, boost::container::vector<std::string> > m_elements;

public:
	CsvDocument()
		: m_elements()
	{
	}
#ifdef POSEIDON_CXX11
	explicit CsvDocument(std::initializer_list<SharedNts> headers)
		: m_elements()
	{
		reset_headers(headers);
	}
#endif
	explicit CsvDocument(std::istream &is)
		: m_elements()
	{
		parse(is);
	}
#ifndef POSEIDON_CXX11
	CsvDocument(const CsvDocument &rhs)
		: m_elements(rhs.m_elements)
	{
	}
	CsvDocument &operator=(const CsvDocument &rhs){
		m_elements = rhs.m_elements;
		return *this;
	}
#endif

public:
	void reset_headers(){
		m_elements.clear();
	}
	void reset_headers(const boost::container::map<SharedNts, std::string> &row){
		VALUE_TYPE(m_elements) elements;
		for(AUTO(it, row.begin()); it != row.end(); ++it){
			elements[it->first];
		}
		m_elements.swap(elements);
	}
#ifdef POSEIDON_CXX11
	void reset_headers(std::initializer_list<SharedNts> headers){
		VALUE_TYPE(m_elements) elements;
		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			elements[*it];
		}
		m_elements.swap(elements);
	}
#endif
	void append(const boost::container::map<SharedNts, std::string> &row){
		AUTO(it, m_elements.begin());
		try {
			while(it != m_elements.end()){
				std::string value;
				const AUTO(rit, row.find(it->first));
				if(rit != row.end()){
					value = rit->second;
				}
				it->second.push_back(STD_MOVE(value));
				++it;
			}
		} catch(...){
			while(it != m_elements.begin()){
				--it;
				it->second.pop_back();
			}
			throw;
		}
	}

	bool empty() const {
		return size() == 0;
	}
	std::size_t size() const {
		if(m_elements.empty()){
			return 0;
		}
		return m_elements.begin()->second.size();
	}
	void clear(){
		for(AUTO(it, m_elements.begin()); it != m_elements.end(); ++it){
			it->second.clear();
		}
	}

	const std::string &get(std::size_t row, const char *key) const { // 若指定的键不存在，则返回空字符串。
		return get(row, SharedNts::view(key));
	}
	const std::string &get(std::size_t row, const SharedNts &column) const {
		const AUTO(it, m_elements.find(column));
		if(it == m_elements.end()){
			return empty_string();
		}
		if(row >= it->second.size()){
			return empty_string();
		}
		return it->second.at(row);
	}
	const std::string &at(std::size_t row, const char *key) const { // 若指定的键不存在，则返回空字符串。
		return at(row, SharedNts::view(key));
	}
	const std::string &at(std::size_t row, const SharedNts &column) const {
		const AUTO(it, m_elements.find(column));
		if(it == m_elements.end()){
			throw std::out_of_range(__PRETTY_FUNCTION__);
		}
		if(row >= it->second.size()){
			throw std::out_of_range(__PRETTY_FUNCTION__);
		}
		return it->second.at(row);
	}

	void swap(CsvDocument &rhs) NOEXCEPT {
		using std::swap;
		swap(m_elements, rhs.m_elements);
	}

	std::string dump() const;
	void dump(std::ostream &os) const;
	void parse(std::istream &is);
};

inline void swap(CsvDocument &lhs, CsvDocument &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

inline std::ostream &operator<<(std::ostream &os, const CsvDocument &rhs){
	rhs.dump(os);
	return os;
}
inline std::istream &operator>>(std::istream &is, CsvDocument &rhs){
	rhs.parse(is);
	return is;
}

}

#endif
