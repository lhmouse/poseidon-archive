// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_OPTIONAL_MAP_HPP_
#define POSEIDON_OPTIONAL_MAP_HPP_

#include "cxx_ver.hpp"
#include <map>
#include "shared_nts.hpp"

namespace Poseidon {

class OptionalMap {
public:
	typedef std::multimap<SharedNts, std::string> delegate_container;

	typedef delegate_container::const_iterator const_iterator;
	typedef delegate_container::iterator iterator;

private:
	delegate_container m_delegate;

public:
	bool empty() const {
		return m_delegate.empty();
	}
	std::size_t size() const {
		return m_delegate.size();
	}
	void clear(){
		m_delegate.clear();
	}

	const_iterator begin() const {
		return m_delegate.begin();
	}
	iterator begin(){
		return m_delegate.begin();
	}
	const_iterator end() const {
		return m_delegate.end();
	}
	iterator end(){
		return m_delegate.end();
	}

	void erase(iterator pos){
		m_delegate.erase(pos);
	}
	void erase(const char *key){
		m_delegate.erase(SharedNts::observe(key));
	}

	void swap(OptionalMap &rhs) NOEXCEPT {
		m_delegate.swap(rhs.m_delegate);
	}

	// 一对一的接口。
	const_iterator find(const char *key) const {
		return m_delegate.find(SharedNts::observe(key));
	}
	iterator find(const char *key){
		return m_delegate.find(SharedNts::observe(key));
	}

	bool has(const char *key) const {
		return find(key) == end();
	}
	iterator create(const char *key){
		iterator ret = find(key);
		if(ret == m_delegate.end()){
			ret = m_delegate.insert(std::make_pair(SharedNts(key), std::string()));
		}
		return ret;
	}
	std::string &set(const char *key, std::string val){
		AUTO_REF(ret, create(key)->second);
		ret.swap(val);
		return ret;
	}

	const std::string &get(const char *key) const; // 若指定的键不存在，则返回空字符串。
	const std::string &at(const char *key) const; // 若指定的键不存在，则抛出 std::out_of_range。

	// 一对多的接口。
	std::pair<const_iterator, const_iterator> range(const char *key) const {
		return m_delegate.equal_range(SharedNts::observe(key));
	}
	std::pair<iterator, iterator> range(const char *key){
		return m_delegate.equal_range(SharedNts::observe(key));
	}
	std::size_t count(const char *key) const {
		return m_delegate.count(SharedNts::observe(key));
	}

	iterator append(const char *key, std::string val){
		return m_delegate.insert(std::make_pair(SharedNts(key), STD_MOVE(val)));
	}
};

}

#endif
