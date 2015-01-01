// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "utilities.hpp"
#include "../utilities.hpp"
using namespace Poseidon;

namespace Poseidon {

std::ostream &operator<<(std::ostream &os, const MySqlStringEscaper &rhs){
	for(AUTO(it, rhs.str.begin()); it != rhs.str.end(); ++it){
		switch(*it){
		case 0:
			os <<'\\' <<'0';
			break;

		case 0x1A:
			os <<'\\' <<'Z';
			break;

		case '\r':
			os <<'\\' <<'r';
			break;

		case '\n':
			os <<'\\' <<'n';
			break;

		case '\'':
			os <<'\\' <<'\'';
			break;

		case '\"':
			os <<'\\' <<'\"';
			break;

		default:
			os <<*it;
			break;
		}
	}
	return os;
}

std::ostream &operator<<(std::ostream &os, const MySqlDateFormatter &rhs){
	char temp[256];
	const std::size_t len = formatTime(temp, sizeof(temp), rhs.time, true);
	return os.write(temp, len);
}

}
