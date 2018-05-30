// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CSV_DOCUMENT_HPP_
#define POSEIDON_CSV_DOCUMENT_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <iosfwd>
#include <stdexcept>
#include <cstddef>
#include <boost/container/vector.hpp>
#include <boost/container/map.hpp>
#include "rcnts.hpp"

namespace Poseidon {

class Stream_buffer;

extern const std::string & empty_string() NOEXCEPT;

class Csv_document {
private:
	boost::container::map<Rcnts, boost::container::vector<std::string> > m_elements;

public:
	Csv_document()
		: m_elements()
	{
		//
	}
#ifdef POSEIDON_CXX11
	explicit Csv_document(std::initializer_list<Rcnts> header)
		: m_elements()
	{
		reset_header(header);
	}
#endif
	explicit Csv_document(std::istream &is);
#ifndef POSEIDON_CXX11
	Csv_document(const Csv_document &rhs)
		: m_elements(rhs.m_elements)
	{
		//
	}
	Csv_document & operator=(const Csv_document &rhs){
		m_elements = rhs.m_elements;
		return *this;
	}
#endif

public:
	void reset_header(){
		m_elements.clear();
	}
	void reset_header(const boost::container::map<Rcnts, std::string> &row);
#ifdef POSEIDON_CXX11
	void reset_header(std::initializer_list<Rcnts> header);
#endif
	void append(const boost::container::map<Rcnts, std::string> &row);
#ifdef POSEIDON_CXX11
	void append(boost::container::map<Rcnts, std::string> &&row);
#endif

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

	const std::string & get(std::size_t row, const char *key) const { // 若指定的键不存在，则返回空字符串。
		return get(row, Rcnts::view(key));
	}
	const std::string & get(std::size_t row, const Rcnts &key) const {
		const AUTO(it, m_elements.find(key));
		if(it == m_elements.end()){
			return empty_string();
		}
		if(row >= it->second.size()){
			return empty_string();
		}
		return it->second.at(row);
	}
	const std::string & at(std::size_t row, const char *key) const { // 若指定的键不存在，则返回空字符串。
		return at(row, Rcnts::view(key));
	}
	const std::string & at(std::size_t row, const Rcnts &key) const {
		const AUTO(it, m_elements.find(key));
		if(it == m_elements.end()){
			throw std::out_of_range(__PRETTY_FUNCTION__);
		}
		if(row >= it->second.size()){
			throw std::out_of_range(__PRETTY_FUNCTION__);
		}
		return it->second.at(row);
	}
	std::string & at(std::size_t row, const char *key){ // 若指定的键不存在，则返回空字符串。
		return at(row, Rcnts::view(key));
	}
	std::string & at(std::size_t row, const Rcnts &key){
		const AUTO(it, m_elements.find(key));
		if(it == m_elements.end()){
			throw std::out_of_range(__PRETTY_FUNCTION__);
		}
		if(row >= it->second.size()){
			throw std::out_of_range(__PRETTY_FUNCTION__);
		}
		return it->second.at(row);
	}

	void swap(Csv_document &rhs) NOEXCEPT {
		using std::swap;
		swap(m_elements, rhs.m_elements);
	}

	Stream_buffer dump() const;
	void dump(std::ostream &os) const;
	void parse(std::istream &is);
};

inline void swap(Csv_document &lhs, Csv_document &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

inline std::ostream & operator<<(std::ostream &os, const Csv_document &rhs){
	rhs.dump(os);
	return os;
}
inline std::istream & operator>>(std::istream &is, Csv_document &rhs){
	rhs.parse(is);
	return is;
}

}

#endif
