// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "optional_map.hpp"
#include <stdexcept>
#include "log.hpp"
using namespace Poseidon;

namespace {

const std::string EMPTY_STRING;

}

const std::string &OptionalMap::get(const SharedNtmbs &key) const {
	const const_iterator it = find(key);
	if(it == end()){
		return EMPTY_STRING;
	}
	return it->second;
}
const std::string &OptionalMap::at(const SharedNtmbs &key) const {
	const const_iterator it = find(key);
	if(it == end()){
		LOG_POSEIDON_WARN("Key not found: ", key);
		throw std::out_of_range("OptionalMap::at");
	}
	return it->second;
}
std::string &OptionalMap::set(const SharedNtmbs &key, std::string val){
	iterator it = create(key);
	it->second.swap(val);
	return it->second;
}
