#include "../precompiled.hpp"
#include "optional_map.hpp"
using namespace Poseidon;

namespace {

const std::string EMPTY_STRING;

void deleteCharArray(char *s){
	delete[] s;
}

}

const std::string &OptionalMap::get(const char *key) const {
	const AUTO(it, m_delegate.find(boost::shared_ptr<const char>(boost::shared_ptr<void>(), key)));
	if(it == m_delegate.end()){
		return EMPTY_STRING;
	}
	return it->second;
}
const std::string &OptionalMap::get(const std::string &key) const {
	return get(key.c_str());
}

std::string &OptionalMap::create(const char *key){
#if __cplusplus >= 201103L
	AUTO(hint, m_delegate.upper_bound(boost::shared_ptr<const char>(boost::shared_ptr<void>(), key)));
	if(hint != m_delegate.begin()){
		AUTO(it, hint);
		--it;
		if(std::strcmp(key, it->first.get()) == 0){
			return it->second;
		}
	}
#else
	AUTO(hint, m_delegate.lower_bound(boost::shared_ptr<const char>(boost::shared_ptr<void>(), key)));
	if(hint != m_delegate.end()){
		if(std::strcmp(key, hint->first.get()) == 0){
			return hint->second;
		}
	}
	if(hint != m_delegate.begin()){
		--hint;
	}
#endif
	const std::size_t size = std::strlen(key) + 1;
	boost::shared_ptr<char> str(new char[size], &deleteCharArray);
	std::memcpy(str.get(), key, size);
	return m_delegate.insert(hint, std::make_pair(str, std::string()))->second;
}
std::string &OptionalMap::create(const std::string &key){
	return create(key.c_str());
}
