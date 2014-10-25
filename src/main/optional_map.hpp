#ifndef POSEIDON_OPTIONAL_MAP_HPP_
#define POSEIDON_OPTIONAL_MAP_HPP_

#include "cxx_ver.hpp"
#include <map>
#include "shared_ntmbs.hpp"

namespace Poseidon {

class OptionalMap {
public:
	typedef std::multimap<SharedNtmbs, std::string> delegate_container;

	typedef delegate_container::const_iterator const_iterator;
	typedef delegate_container::iterator iterator;

private:
	static const std::string EMPTY_STRING;

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
	void erase(const SharedNtmbs &key){
		m_delegate.erase(key);
	}

	void swap(OptionalMap &rhs) NOEXCEPT {
		m_delegate.swap(rhs.m_delegate);
	}

	// 一对一的接口。
	const_iterator find(const SharedNtmbs &key) const {
		return m_delegate.find(key);
	}
	iterator find(const SharedNtmbs &key){
		return m_delegate.find(key);
	}

	bool has(const SharedNtmbs &key) const {
		return find(key) == end();
	}
	iterator create(const SharedNtmbs &key){
		iterator ret = m_delegate.find(key);
		if(ret == m_delegate.end()){
			ret = m_delegate.insert(std::make_pair(SharedNtmbs(key, true), std::string()));
		}
		return ret;
	}

	const std::string &get(const SharedNtmbs &key) const {
		const const_iterator it = find(key);
		return (it != end()) ? it->second : EMPTY_STRING;
	}
	std::string &set(const SharedNtmbs &key, std::string val){
		iterator it = create(key);
		it->second.swap(val);
		return it->second;
	}

	// 一对多的接口。
	std::pair<const_iterator, const_iterator> range(const SharedNtmbs &key) const {
		return m_delegate.equal_range(key);
	}
	std::pair<iterator, iterator> range(const SharedNtmbs &key){
		return m_delegate.equal_range(key);
	}
	std::size_t count(const SharedNtmbs &key) const {
		return m_delegate.count(key);
	}

	iterator append(const SharedNtmbs &key, std::string val){
		return m_delegate.insert(std::make_pair(SharedNtmbs(key, true), STD_MOVE(val)));
	}
};

}

#endif
