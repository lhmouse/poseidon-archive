#ifndef POSEIDON_OPTIONAL_MAP_HPP_
#define POSEIDON_OPTIONAL_MAP_HPP_

#include <map>
#include <string>

namespace Poseidon {

namespace {
	static std::string EMPTY_STRING;
}

class OptionalMap
	: public std::map<std::string, std::string>
{
public:
	const std::string &get(const std::string &key) const {
		const std::map<std::string, std::string>::const_iterator it = find(key);
		if(it == end()){
			return EMPTY_STRING;
		}
		return it->second;
	}
	void set(std::string key, std::string val){
		std::map<std::string, std::string>::operator[](key).swap(val);
	}

	const std::string &operator[](const std::string &key) const {
		return get(key);
	}
};

}

#endif
