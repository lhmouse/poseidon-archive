// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_UTILITIES_HPP_
#define POSEIDON_MYSQL_UTILITIES_HPP_

#include <string>

namespace Poseidon {

extern std::string escapeStringForSql(std::string str);

}

#endif
