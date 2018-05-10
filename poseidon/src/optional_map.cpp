// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "optional_map.hpp"

namespace Poseidon {

std::ostream &operator<<(std::ostream &os, const Optional_map &rhs){
	os <<"{\n";
	for(AUTO(it, rhs.begin()); it != rhs.end(); ++it){
		os <<"  " <<it->first <<": string(" <<it->second.size() <<") = " <<it->second <<"\n";
	}
	os <<"}\n";
	return os;
}

}
