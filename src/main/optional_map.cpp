// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "optional_map.hpp"
using namespace Poseidon;

namespace {

const std::string EMPTY_STRING;

}

const std::string &OptionalMap::get(const SharedNtmbs &key) const {
	const const_iterator it = find(key);
	return (it != end()) ? it->second : EMPTY_STRING;
}
std::string &OptionalMap::set(const SharedNtmbs &key, std::string val){
	iterator it = create(key);
	it->second.swap(val);
	return it->second;
}
