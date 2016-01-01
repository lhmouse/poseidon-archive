// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "utilities.hpp"
#include "../time.hpp"

namespace Poseidon {

namespace MySql {
	std::ostream &operator<<(std::ostream &os, const StringEscaper &rhs){
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
		return os;
	}

	std::ostream &operator<<(std::ostream &os, const DateFormatter &rhs){
		char temp[256];
		const std::size_t len = format_time(temp, sizeof(temp), rhs.time, true);
		return os.write(temp, static_cast<std::streamsize>(len));
	}
}

}
