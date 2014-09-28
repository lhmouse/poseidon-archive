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

std::string &OptionalMap::create(const char *key){
	AUTO(range, m_delegate.equal_range(
		boost::shared_ptr<const char>(boost::shared_ptr<void>(), key)));
	AUTO(it, range.first);
	if(it != range.second){
		++range.first;
		m_delegate.erase(range.first, range.second);
		return it->second;
	}
	const std::size_t len = std::strlen(key);
	boost::shared_ptr<char> newKey(new char[len + 1], &deleteCharArray);
	std::memcpy(newKey.get(), key, len + 1);
#ifdef POSEIDON_CXX11
	return m_delegate.insert(range.second,
#else
	if(range.first != m_delegate.begin()){
		--range.first;
	}
	return m_delegate.insert(range.first,
#endif
		std::make_pair(newKey, EMPTY_STRING))->second;
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
