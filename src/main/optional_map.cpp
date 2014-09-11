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
	const AUTO(it, m_delegate.find(
		boost::shared_ptr<const char>(boost::shared_ptr<void>(), key)));
	if(it == m_delegate.end()){
		return EMPTY_STRING;
	}
	return it->second;
}

OptionalMap::iterator OptionalMap::create(const char *key, std::size_t len){
#if __cplusplus >= 201103L
	AUTO(hint, m_delegate.upper_bound(
		boost::shared_ptr<const char>(boost::shared_ptr<void>(), key)));
	if(hint != m_delegate.begin()){
		AUTO(it, hint);
		--it;
		if(std::strcmp(key, it->first.get()) == 0){
			return it;
		}
	}
#else
	AUTO(hint, m_delegate.lower_bound(
		boost::shared_ptr<const char>(boost::shared_ptr<void>(), key)));
	if(hint != m_delegate.end()){
		if(std::strcmp(key, hint->first.get()) == 0){
			return hint;
		}
	}
	if(hint != m_delegate.begin()){
		--hint;
	}
#endif
	boost::shared_ptr<char> str(new char[len + 1], &deleteCharArray);
	std::memcpy(str.get(), key, len);
	str.get()[len] = 0;
	return m_delegate.insert(hint, std::make_pair(str, EMPTY_STRING));
}

std::pair<OptionalMap::const_iterator, OptionalMap::const_iterator>
	OptionalMap::range(const char *key) const
{
	return m_delegate.equal_range(
		boost::shared_ptr<const char>(boost::shared_ptr<void>(), key));
}
std::size_t OptionalMap::count(const char *key) const {
	return m_delegate.count(
		boost::shared_ptr<const char>(boost::shared_ptr<void>(), key));
}

OptionalMap::iterator OptionalMap::add(const char *key, std::size_t len, std::string val){
	boost::shared_ptr<char> str(new char[len + 1], &deleteCharArray);
	std::memcpy(str.get(), key, len);
	str.get()[len] = 0;
	return m_delegate.insert(std::make_pair(str, val));
}
