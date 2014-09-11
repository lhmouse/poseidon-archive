#ifndef POSEIDON_OPTIONAL_MAP_HPP_
#define POSEIDON_OPTIONAL_MAP_HPP_

#include "../cxx_ver.hpp"
#include <map>
#include <string>
#include <cstring>
#include <cstddef>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class OptionalMap {
public:
	struct Comparator {
		bool operator()(const boost::shared_ptr<const char> &lhs,
			const boost::shared_ptr<const char> &rhs) const
		{
			return std::strcmp(lhs.get(), rhs.get()) < 0;
		}
	};

	typedef std::multimap<
		boost::shared_ptr<const char>, std::string,
		Comparator
		> delegate_container;

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

#ifdef POSEIDON_CXX11
	void erase(const_iterator pos){
		m_delegate.erase(pos);
	}
#endif
	void erase(iterator pos){
		m_delegate.erase(pos);
	}

	void swap(OptionalMap &rhs) NOEXCEPT {
		m_delegate.swap(rhs.m_delegate);
	}

	// 一对一的接口。
	const std::string &get(const char *key) const;
	const std::string &get(const std::string &key) const {
		return get(key.c_str());
	}

	const std::string &operator[](const std::string &key) const {
		return get(key);
	}

	iterator create(const char *key, std::size_t len);
	iterator create(const char *key){
		return create(key, std::strlen(key));
	}
	iterator create(const std::string &key){
		return create(key.data(), key.size());
	}

	iterator set(const char *key, std::size_t len, std::string val){
		iterator ret = create(key, len);
		val.swap(ret->second);
		return STD_MOVE(ret);
	}
	iterator set(const char *key, std::string val){
		iterator ret = create(key);
		val.swap(ret->second);
		return STD_MOVE(ret);
	}
	iterator set(const std::string &key, std::string val){
		iterator ret = create(key);
		val.swap(ret->second);
		return STD_MOVE(ret);
	}

	// 一对多的接口。
	std::pair<const_iterator, const_iterator> range(const char *key) const;
	std::pair<const_iterator, const_iterator> range(const std::string &key) const {
		return range(key.c_str());
	}

	std::size_t count(const char *key) const;
	std::size_t count(const std::string &key) const {
		return count(key.c_str());
	}

	iterator add(const char *key, std::size_t len, std::string val);
	iterator add(const char *key, std::string val){
		return add(key, std::strlen(key), STD_MOVE(val));
	}
	iterator add(const std::string &key, std::string val){
		return add(key.data(), key.size(), STD_MOVE(val));
	}
};

}

#endif
