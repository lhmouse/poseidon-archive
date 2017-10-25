// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_OPTIONAL_MAP_HPP_
#define POSEIDON_OPTIONAL_MAP_HPP_

#include "cxx_ver.hpp"
#include <boost/container/map.hpp>
#include <stdexcept>
#include <iosfwd>
#include "shared_nts.hpp"

namespace Poseidon {

extern const std::string &empty_string() NOEXCEPT;

class OptionalMap {
public:
	typedef boost::container::multimap<SharedNts, std::string> base_container;

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
	OptionalMap()
		: m_elements()
	{ }
#ifndef POSEIDON_CXX11
	OptionalMap(const OptionalMap &rhs)
		: m_elements(rhs.m_elements)
	{ }
	OptionalMap &operator=(const OptionalMap &rhs){
		m_elements = rhs.m_elements;
		return *this;
	}
#endif
	~OptionalMap();

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
#ifdef POSEIDON_CXX11
	const_iterator cbegin() const {
		return m_elements.begin();
	}
#endif
	const_iterator end() const {
		return m_elements.end();
	}
	iterator end(){
		return m_elements.end();
	}
#ifdef POSEIDON_CXX11
	const_iterator cend() const {
		return m_elements.end();
	}
#endif

	const_reverse_iterator rbegin() const {
		return m_elements.rbegin();
	}
	reverse_iterator rbegin(){
		return m_elements.rbegin();
	}
#ifdef POSEIDON_CXX11
	const_reverse_iterator crbegin() const {
		return m_elements.rbegin();
	}
#endif
	const_reverse_iterator rend() const {
		return m_elements.rend();
	}
	reverse_iterator rend(){
		return m_elements.rend();
	}
#ifdef POSEIDON_CXX11
	const_reverse_iterator crend() const {
		return m_elements.rend();
	}
#endif

	iterator erase(const_iterator pos){
		return m_elements.erase(pos);
	}
	iterator erase(const_iterator first, const_iterator last){
		return m_elements.erase(first, last);
	}
	size_type erase(const char *key){
		return erase(SharedNts::view(key));
	}
	size_type erase(const SharedNts &key){
		return m_elements.erase(key);
	}

	void swap(OptionalMap &rhs) NOEXCEPT {
		using std::swap;
		swap(m_elements, rhs.m_elements);
	}

	// 一对一的接口。
	const_iterator find(const char *key) const {
		return find(SharedNts::view(key));
	}
	const_iterator find(const SharedNts &key) const {
		return m_elements.find(key);
	}
	iterator find(const char *key){
		return find(SharedNts::view(key));
	}
	iterator find(const SharedNts &key){
		return m_elements.find(key);
	}

	bool has(const char *key) const {
		return find(key) != end();
	}
	bool has(const SharedNts &key){
		return find(key) != end();
	}
	iterator set(SharedNts key, std::string val){
		AUTO(range, m_elements.equal_range(key));
		if(range.first == range.second){
			return m_elements.emplace(STD_MOVE_IDN(key), STD_MOVE_IDN(val));
		} else {
			range.first = m_elements.erase(range.first, --range.second);
			range.first->second.swap(val);
			return range.first;
		}
	}

	const std::string &get(const char *key) const { // 若指定的键不存在，则返回空字符串。
		return get(SharedNts::view(key));
	};
	const std::string &get(const SharedNts &key) const {
		const AUTO(it, find(key));
		if(it == end()){
			return empty_string();
		}
		return it->second;
	}
	const std::string &at(const char *key) const { // 若指定的键不存在，则抛出 std::out_of_range。
		return at(SharedNts::view(key));
	};
	const std::string &at(const SharedNts &key) const {
		const AUTO(it, find(key));
		if(it == end()){
			throw std::out_of_range(__PRETTY_FUNCTION__);
		}
		return it->second;
	}
	std::string &at(const char *key){ // 若指定的键不存在，则抛出 std::out_of_range。
		return at(SharedNts::view(key));
	};
	std::string &at(const SharedNts &key){
		const AUTO(it, find(key));
		if(it == end()){
			throw std::out_of_range(__PRETTY_FUNCTION__);
		}
		return it->second;
	}

	// 一对多的接口。
	std::pair<const_iterator, const_iterator> range(const char *key) const {
		return range(SharedNts::view(key));
	}
	std::pair<const_iterator, const_iterator> range(const SharedNts &key) const {
		return m_elements.equal_range(key);
	}
	std::pair<iterator, iterator> range(const char *key){
		return range(SharedNts::view(key));
	}
	std::pair<iterator, iterator> range(const SharedNts &key){
		return m_elements.equal_range(key);
	}
	size_type count(const char *key) const {
		return count(SharedNts::view(key));
	}
	size_type count(const SharedNts &key) const {
		return m_elements.count(key);
	}

	iterator append(SharedNts key, std::string val){
		return m_elements.emplace(STD_MOVE_IDN(key), STD_MOVE_IDN(val));
	}
};

inline void swap(OptionalMap &lhs, OptionalMap &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

extern std::ostream &operator<<(std::ostream &os, const OptionalMap &rhs);

}

#endif
