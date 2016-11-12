// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_OPTIONAL_MAP_HPP_
#define POSEIDON_OPTIONAL_MAP_HPP_

#include "cxx_ver.hpp"
#include <map>
#include "shared_nts.hpp"

namespace Poseidon {

class OptionalMap {
public:
	typedef std::multimap<SharedNts, std::string> base_container;

	typedef base_container::const_iterator const_iterator;
	typedef base_container::iterator iterator;

private:
	base_container m_elements;

public:
	bool empty() const {
		return m_elements.empty();
	}
	std::size_t size() const {
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
	const_iterator end() const {
		return m_elements.end();
	}
	iterator end(){
		return m_elements.end();
	}

	iterator erase(iterator pos){
		m_elements.erase(pos++);
		return pos;
	}
	std::size_t erase(const char *key){
		return erase(SharedNts::view(key));
	}
	std::size_t erase(const SharedNts &key){
		return m_elements.erase(key);
	}

	void swap(OptionalMap &rhs) NOEXCEPT {
		m_elements.swap(rhs.m_elements);
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
	iterator create(SharedNts key){
		iterator ret = find(key);
		if(ret == m_elements.end()){
			ret = m_elements.insert(std::make_pair(STD_MOVE(key), std::string()));
		}
		return ret;
	}
	std::string &set(SharedNts key, std::string val){
		AUTO_REF(ret, create(STD_MOVE(key))->second);
		ret = STD_MOVE(val);
		return ret;
	}

	const std::string &get(const char *key) const; // 若指定的键不存在，则返回空字符串。
	const std::string &get(const SharedNts &key) const {
		return get(key.get());
	}
	const std::string &at(const char *key) const; // 若指定的键不存在，则抛出 std::out_of_range。
	const std::string &at(const SharedNts &key) const {
		return at(key.get());
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
	std::size_t count(const char *key) const {
		return count(SharedNts::view(key));
	}
	std::size_t count(const SharedNts &key) const {
		return m_elements.count(key);
	}

	iterator append(SharedNts key, std::string val){
		return m_elements.insert(std::make_pair(STD_MOVE(key), STD_MOVE(val)));
	}
};

}

#endif
