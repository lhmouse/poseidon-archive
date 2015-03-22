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

		Request(Verb verb_, std::string uri_, unsigned version_,
			OptionalMap getParams_, OptionalMap headers_, std::string contents_)
			: verb(verb_), uri(STD_MOVE(uri_)), version(version_)
			, getParams(STD_MOVE(getParams_)), headers(STD_MOVE(headers_)), contents(STD_MOVE(contents_))
		{
		}
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
