#ifndef POSEIDON_HTTP_REQUEST_HPP_
#define POSEIDON_HTTP_REQUEST_HPP_

#include "../../cxx_ver.hpp"
#include <string>
#include "verb.hpp"
#include "../optional_map.hpp"

namespace Poseidon {

struct HttpRequest {
	HttpVerb verb;
	std::string uri;
	OptionalMap getParams;
	OptionalMap headers;
	std::string contents;
};

static inline void swap(HttpRequest &lhs, HttpRequest &rhs) NOEXCEPT {
	using std::swap;
	swap(lhs.verb, rhs.verb);
	swap(lhs.uri, rhs.uri);
	swap(lhs.getParams, rhs.getParams);
	swap(lhs.headers, rhs.headers);
	swap(lhs.contents, rhs.contents);
}

}

#endif
