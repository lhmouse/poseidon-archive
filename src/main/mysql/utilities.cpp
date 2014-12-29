// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "utilities.hpp"
#include "../utilities.hpp"
using namespace Poseidon;

namespace Poseidon {

void quoteStringForSql(std::ostream &os, const std::string &str){
	for(AUTO(it, str.begin()); it != str.end(); ++it){
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
}

void formatDateTime(std::ostream &os, double datetime){
	char temp[256];
	const AUTO(len, formatTime(temp, sizeof(temp), datetime * (24 * 3600 * 1000), true));
	os.write(temp, len);
}
double scanDateTime(const char *str){
	return static_cast<double>(scanTime(str)) / (24 * 3600 * 1000);
}

}
