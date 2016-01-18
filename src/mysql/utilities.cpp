// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "utilities.hpp"
#include "../time.hpp"
#include "../uuid.hpp"

namespace Poseidon {

namespace MySql {
	std::ostream &operator<<(std::ostream &os, const StringEscaper &rhs){
		os <<'\'';
		for(AUTO(it, rhs.str.begin()); it != rhs.str.end(); ++it){
			switch(*it){
			case 0:
				os.put('\\').put('0');
				break;
			case 0x1A:
				os.put('\\').put('Z');
				break;
			case '\r':
				os.put('\\').put('r');
				break;
			case '\n':
				os.put('\\').put('n');
				break;
			case '\\':
				os.put('\\').put('\\');
				break;
			case '\'':
				os.put('\\').put('\'');
				break;
			case '\"':
				os.put('\\').put('\"');
				break;
			default:
				os.put(*it);
				break;
			}
		}
		os <<'\'';
		return os;
	}

	std::ostream &operator<<(std::ostream &os, const DateTimeFormatter &rhs){
		char str[256];
		format_time(str, sizeof(str), rhs.time, true);
		os <<'\'';
		os <<str;
		os <<'\'';
		return os;
	}

	std::ostream &operator<<(std::ostream &os, const UuidFormatter &rhs){
		os <<'\'';
		os <<rhs.uuid;
		os <<'\'';
		return os;
	}
}

}
