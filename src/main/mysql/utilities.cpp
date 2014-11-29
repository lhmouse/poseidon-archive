// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "utilities.hpp"
using namespace Poseidon;

namespace Poseidon {

std::string escapeStringForSql(std::string str){
	std::string ret;
	ret.swap(str);
	std::size_t i = 0;
	while(i < str.size()){
		switch(str[i]){
		case 0:
			str.replace(i, 1, "\\0");
			++i;
			break;

		case 0x1A:
			str.replace(i, 1, "\\Z");
			++i;
			break;

		case '\r':
			str.replace(i, 1, "\\r");
			++i;
			break;

		case '\n':
			str.replace(i, 1, "\\n");
			++i;
			break;

		case '\'':
			str.replace(i, 1, "\\\'");
			++i;
			break;

		case '\"':
			str.replace(i, 1, "\\\"");
			++i;
			break;
		}
		++i;
	}
	return ret;
}

}
