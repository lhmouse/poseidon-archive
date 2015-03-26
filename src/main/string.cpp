// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "string.hpp"
#include <iomanip>

namespace Poseidon {

std::ostream &operator<<(std::ostream &os, const HexDumper &dumper){
	AUTO(read, static_cast<const unsigned char *>(dumper.read));
	os <<std::hex;
	for(std::size_t i = 0; i < dumper.size; ++i){
		os <<std::setfill('0') <<std::setw(2) <<static_cast<unsigned>(*read) <<' ';
		++read;
	}
	os << std::dec;
	return os;
}

}
