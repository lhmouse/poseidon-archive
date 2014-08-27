#ifndef POSEIDON_OPTIONAL_MAP_HPP_
#define POSEIDON_OPTIONAL_MAP_HPP_

#include <map>
#include <string>
#include <cstddef>

namespace Poseidon {

const std::string EMPTY_STRING;

class OptionalMap {
public:
	typedef std::map<std::string, std::string> delegate_container;

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

	const std::string &get(const std::string &key) const {
		delegate_container::const_iterator it = m_delegate.find(key);
		if(it == m_delegate.end()){
			return EMPTY_STRING;
		}
		return it->second;
	}
	std::string &get(const std::string &key){
		return m_delegate[key];
	}
	void set(const std::string &key, std::string val){
		m_delegate[key].swap(val);
	}

	const std::string &operator[](const std::string &key) const {
		return get(key);
	}
	std::string &operator[](const std::string &key){
		return get(key);
	}
};

}

#endif
