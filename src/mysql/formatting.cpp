// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "formatting.hpp"
#include "../time.hpp"
#include "../uuid.hpp"

namespace Poseidon {
namespace MySql {

std::ostream &operator<<(std::ostream &os, const StringEscaper &rhs){
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

std::ostream &operator<<(std::ostream &os, const DateTimeFormatter &rhs){
	const AUTO_REF(ref, rhs.get());
	char str[256];
	format_time(str, sizeof(str), ref, true);
	os <<'\'';
	os <<str;
	os <<'\'';
	return os;
}

std::ostream &operator<<(std::ostream &os, const UuidFormatter &rhs){
	const AUTO_REF(ref, rhs.get());
	os <<'\'';
	os <<ref;
	os <<'\'';
	return os;
}

}
}
