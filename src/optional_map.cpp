// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "optional_map.hpp"

namespace Poseidon {

const std::string &OptionalMap::get(const SharedNts &key) const {
	const AUTO(it, find(key));
	if(it == end()){
		return EMPTY_STRING;
	}
	return it->second;
}
const std::string &OptionalMap::at(const SharedNts &key) const {
	const AUTO(it, find(key));
	if(it == end()){
		throw std::out_of_range(__PRETTY_FUNCTION__);
	}
	return it->second;
}

}
