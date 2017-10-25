// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "hex_printer.hpp"

namespace Poseidon {

std::ostream &operator<<(std::ostream &os, const HexPrinter &rhs){
	static CONSTEXPR const char HEX_TABLE[] = "0123456789abcdef";
	AUTO(read, static_cast<const unsigned char *>(rhs.get_data()));
	for(std::size_t i = 0; i < rhs.get_size(); ++i){
		const unsigned byte = *(read++);
		os <<HEX_TABLE[byte / 16] <<HEX_TABLE[byte % 16] <<rhs.get_delimiter();
	}
	return os;
}

}
