#ifndef POSEIDON_OPTIONAL_MAP_HPP_
#define POSEIDON_OPTIONAL_MAP_HPP_

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

	typedef std::map<
		boost::shared_ptr<const char>, std::string,
		Comparator
		> delegate_container;

	typedef delegate_container::const_iterator const_iterator;
	typedef delegate_container::iterator iterator;

private:
	delegate_container m_delegate;

public:
	const std::string &get(const char *key) const;
	const std::string &get(const std::string &key) const;

	std::string &create(const char *key);
	std::string &create(const std::string &key);

	void set(const char *key, std::string val){
		create(key).swap(val);
	}
	void set(const std::string &key, std::string val){
		create(key).swap(val);
	}

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

	void swap(OptionalMap &rhs){
		m_delegate.swap(rhs.m_delegate);
	}

	const std::string &operator[](const std::string &key) const {
		return get(key);
	}
	const std::string &operator[](const std::string &key){
		return get(key);
	}
};

}

#endif
