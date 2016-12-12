// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_URLENCODED_HPP_
#define POSEIDON_HTTP_URLENCODED_HPP_

#include <string>
#include <iosfwd>
#include "../optional_map.hpp"

namespace Poseidon {

namespace Http {
	extern void url_encode(std::ostream &os, const std::string &str);
	extern void url_decode(std::istream &is, std::string &str);

	extern void url_encode_params(std::ostream &os, const OptionalMap &params);
	extern void url_decode_params(std::istream &is, OptionalMap &params);
}

}

#endif
