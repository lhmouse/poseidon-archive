// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_REQUEST_HPP_
#define POSEIDON_HTTP_REQUEST_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include "verbs.hpp"
#include "../optional_map.hpp"

namespace Poseidon {

namespace Http {
	struct Request {
		Verb verb;
		std::string uri;
		unsigned version;
		OptionalMap getParams;
		OptionalMap headers;
		std::string contents;
	};

	inline void swap(Request &lhs, Request &rhs) NOEXCEPT {
		using std::swap;
		swap(lhs.verb, rhs.verb);
		swap(lhs.uri, rhs.uri);
		swap(lhs.version, rhs.version);
		swap(lhs.getParams, rhs.getParams);
		swap(lhs.headers, rhs.headers);
		swap(lhs.contents, rhs.contents);
	}
}

}

#endif
