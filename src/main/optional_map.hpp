#ifndef POSEIDON_OPTIONAL_MAP_HPP_
#define POSEIDON_OPTIONAL_MAP_HPP_

#include "../cxx_ver.hpp"
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
	void erase(const char *key){
		m_delegate.erase(SharedNtmbs::createNonOwning(key));
	}
	void erase(const std::string &key){
		m_delegate.erase(SharedNtmbs::createNonOwning(key));
	}

	void swap(OptionalMap &rhs) NOEXCEPT {
		m_delegate.swap(rhs.m_delegate);
	}

	// 一对一的接口。
	const std::string &get(const SharedNtmbs &key) const {
		const_iterator it = m_delegate.find(key);
		if(it == m_delegate.end()){
			return EMPTY_STRING;
		}
		return it->second;
	}
	const std::string &get(const char *key) const {
		return get(SharedNtmbs::createNonOwning(key));
	}
	const std::string &get(const std::string &key) const {
		return get(SharedNtmbs::createNonOwning(key));
	}

	bool hasKey(const SharedNtmbs &key) const {
		return m_delegate.find(key) != m_delegate.end();
	}
	bool hasKey(const char *key) const {
		return hasKey(SharedNtmbs::createNonOwning(key));
	}
	bool hasKey(const std::string &key) const {
		return hasKey(SharedNtmbs::createNonOwning(key));
	}

	std::string &create(const SharedNtmbs &key){
		iterator it = m_delegate.find(key);
		if(it == m_delegate.end()){
			it = m_delegate.insert(it, std::make_pair(key.forkOwning(), std::string()));
		}
		return it->second;
	}
	std::string &create(const char *key){
		return create(SharedNtmbs::createNonOwning(key));
	}
	std::string &create(const std::string &key){
		return create(SharedNtmbs::createNonOwning(key));
	}

	std::string &set(const SharedNtmbs &key, std::string val){
		std::string &ret = create(key);
		ret.swap(val);
		return ret;
	}
	std::string &set(const char *key, std::string val){
		std::string &ret = create(key);
		ret.swap(val);
		return ret;
	}
	std::string &set(const std::string &key, std::string val){
		std::string &ret = create(key);
		ret.swap(val);
		return ret;
	}

	// 一对多的接口。
	std::pair<const_iterator, const_iterator> range(const SharedNtmbs &key) const {
		return m_delegate.equal_range(key);
	}
	std::pair<const_iterator, const_iterator> range(const char *key) const {
		return range(SharedNtmbs::createNonOwning(key));
	}
	std::pair<const_iterator, const_iterator> range(const std::string &key) const {
		return range(SharedNtmbs::createNonOwning(key));
	}

	std::size_t count(const SharedNtmbs &key) const {
		return m_delegate.count(key);
	}
	std::size_t count(const char *key) const {
		return count(SharedNtmbs::createNonOwning(key));
	}
	std::size_t count(const std::string &key) const {
		return count(SharedNtmbs::createNonOwning(key));
	}

	iterator append(const SharedNtmbs &key, std::string val){
		return m_delegate.insert(m_delegate.end(),
			std::make_pair(key.forkOwning(), STD_MOVE(val)));
	}
	iterator append(const char *key, std::string val){
		return m_delegate.insert(m_delegate.end(),
			std::make_pair(SharedNtmbs::createNonOwning(key), STD_MOVE(val)));
	}
	iterator append(const std::string &key, std::string val){
		return m_delegate.insert(m_delegate.end(),
			std::make_pair(SharedNtmbs::createNonOwning(key), STD_MOVE(val)));
	}
};

}

#endif
