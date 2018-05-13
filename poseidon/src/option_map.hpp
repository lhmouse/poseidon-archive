// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_OPTION_MAP_HPP_
#define POSEIDON_OPTION_MAP_HPP_

#include "cxx_ver.hpp"
#include <boost/container/map.hpp>
#include <stdexcept>
#include <iosfwd>
#include "rcnts.hpp"

namespace Poseidon {

extern const std::string &empty_string() NOEXCEPT;

class Option_map {
public:
	typedef boost::container::multimap<Rcnts, std::string> base_container;

	typedef base_container::value_type        value_type;
	typedef base_container::const_reference   const_reference;
	typedef base_container::reference         reference;
	typedef base_container::size_type         size_type;
	typedef base_container::difference_type   difference_type;

	typedef base_container::const_iterator          const_iterator;
	typedef base_container::iterator                iterator;
	typedef base_container::const_reverse_iterator  const_reverse_iterator;
	typedef base_container::reverse_iterator        reverse_iterator;

private:
	base_container m_elements;

public:
	Option_map()
		: m_elements()
	{
		//
	}
#ifndef POSEIDON_CXX11
	Option_map(const Option_map &rhs)
		: m_elements(rhs.m_elements)
	{
		//
	}
	Option_map &operator=(const Option_map &rhs){
		m_elements = rhs.m_elements;
		return *this;
	}
#endif

public:
	bool empty() const {
		return m_elements.empty();
	}
	size_type size() const {
		return m_elements.size();
	}
	void clear(){
		m_elements.clear();
	}

	const_iterator begin() const {
		return m_elements.begin();
	}
	iterator begin(){
		return m_elements.begin();
	}
	const_iterator cbegin() const {
		return m_elements.begin();
	}
	const_iterator end() const {
		return m_elements.end();
	}
	iterator end(){
		return m_elements.end();
	}
	const_iterator cend() const {
		return m_elements.end();
	}

	const_reverse_iterator rbegin() const {
		return m_elements.rbegin();
	}
	reverse_iterator rbegin(){
		return m_elements.rbegin();
	}
	const_reverse_iterator crbegin() const {
		return m_elements.rbegin();
	}
	const_reverse_iterator rend() const {
		return m_elements.rend();
	}
	reverse_iterator rend(){
		return m_elements.rend();
	}
	const_reverse_iterator crend() const {
		return m_elements.rend();
	}

	iterator erase(const_iterator pos){
		return m_elements.erase(pos);
	}
	iterator erase(const_iterator first, const_iterator last){
		return m_elements.erase(first, last);
	}
	size_type erase(const char *key){
		return erase(Rcnts::view(key));
	}
	size_type erase(const Rcnts &key){
		return m_elements.erase(key);
	}

	void swap(Option_map &rhs) NOEXCEPT {
		using std::swap;
		swap(m_elements, rhs.m_elements);
	}

	// 一对一的接口。
	const_iterator find(const char *key) const {
		return find(Rcnts::view(key));
	}
	const_iterator find(const Rcnts &key) const {
		return m_elements.find(key);
	}
	iterator find(const char *key){
		return find(Rcnts::view(key));
	}
	iterator find(const Rcnts &key){
		return m_elements.find(key);
	}

	bool has(const char *key) const {
		return find(key) != end();
	}
	bool has(const Rcnts &key){
		return find(key) != end();
	}
	iterator set(Rcnts key, std::string val){
		AUTO(pair, m_elements.equal_range(key));
		if(pair.first == pair.second){
			return m_elements.emplace(STD_MOVE_IDN(key), STD_MOVE_IDN(val));
		} else {
			pair.first = m_elements.erase(pair.first, --pair.second);
			pair.first->second.swap(val);
			return pair.first;
		}
	}

	const std::string &get(const char *key) const { // 若指定的键不存在，则返回空字符串。
		return get(Rcnts::view(key));
	};
	const std::string &get(const Rcnts &key) const {
		const AUTO(it, find(key));
		if(it == end()){
			return empty_string();
		}
		return it->second;
	}
	const std::string &at(const char *key) const { // 若指定的键不存在，则抛出 std::out_of_range。
		return at(Rcnts::view(key));
	};
	const std::string &at(const Rcnts &key) const {
		const AUTO(it, find(key));
		if(it == end()){
			throw std::out_of_range(__PRETTY_FUNCTION__);
		}
		return it->second;
	}
	std::string &at(const char *key){ // 若指定的键不存在，则抛出 std::out_of_range。
		return at(Rcnts::view(key));
	};
	std::string &at(const Rcnts &key){
		const AUTO(it, find(key));
		if(it == end()){
			throw std::out_of_range(__PRETTY_FUNCTION__);
		}
		return it->second;
	}

	// 一对多的接口。
	std::pair<const_iterator, const_iterator> range(const char *key) const {
		return range(Rcnts::view(key));
	}
	std::pair<const_iterator, const_iterator> range(const Rcnts &key) const {
		return m_elements.equal_range(key);
	}
	std::pair<iterator, iterator> range(const char *key){
		return range(Rcnts::view(key));
	}
	std::pair<iterator, iterator> range(const Rcnts &key){
		return m_elements.equal_range(key);
	}
	size_type count(const char *key) const {
		return count(Rcnts::view(key));
	}
	size_type count(const Rcnts &key) const {
		return m_elements.count(key);
	}

	iterator append(Rcnts key, std::string val){
		return m_elements.emplace(STD_MOVE_IDN(key), STD_MOVE_IDN(val));
	}
};

inline void swap(Option_map &lhs, Option_map &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

extern std::ostream &operator<<(std::ostream &os, const Option_map &rhs);

}

#endif
