// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "formatting.hpp"
#include "../time.hpp"

namespace Poseidon {
namespace Mysql {

std::ostream & operator<<(std::ostream &os, const String_escaper &rhs){
	const AUTO_REF(ref, rhs.get());
	os <<'\'';
	for(AUTO(it, ref.begin()); it != ref.end(); ++it){
		const char ch = *it;
		switch(ch){
		case 0:
			os <<"\\0";
			break;
		case 0x1A:
			os <<"\\Z";
			break;
		case '\r':
			os <<"\\r";
			break;
		case '\n':
			os <<"\\n";
			break;
		case '\\':
			os <<"\\\\";
			break;
		case '\'':
			os <<"\\\'";
			break;
		case '\"':
			os <<"\\\"";
			break;
		default:
			os <<ch;
			break;
		}
	}
	os <<'\'';
	return os;
}

std::ostream & operator<<(std::ostream &os, const Date_time_formatter &rhs){
	const AUTO_REF(ref, rhs.get());
	char str[256];
	format_time(str, sizeof(str), ref, true);
	os <<'\'';
	os <<str;
	os <<'\'';
	return os;
}

std::ostream & operator<<(std::ostream &os, const Uuid_formatter &rhs){
	const AUTO_REF(ref, rhs.get());
	os <<'\'';
	os <<ref;
	os <<'\'';
	return os;
}

}
}
