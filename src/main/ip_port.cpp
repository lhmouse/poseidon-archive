// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "cxx_ver.hpp"
#include "ip_port.hpp"
#include <ostream>
using namespace Poseidon;

namespace Poseidon {

std::ostream &operator<<(std::ostream &os, const IpPort &rhs){
	return os <<rhs.ip <<':' <<rhs.port;
}
std::wostream &operator<<(std::wostream &os, const IpPort &rhs){
	return os <<rhs.ip <<':' <<rhs.port;
}

}
