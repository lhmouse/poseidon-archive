// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_VERB_HPP_
#define POSEIDON_HTTP_VERB_HPP_

namespace Poseidon {

enum HttpVerb {
	HTTP_INVALID_VERB,
	HTTP_GET,
	HTTP_POST,
	HTTP_HEAD,
	HTTP_PUT,
	HTTP_DELETE,
	HTTP_TRACE,
	HTTP_CONNECT,
	HTTP_OPTIONS
};

HttpVerb httpVerbFromString(const char *str);
const char *stringFromHttpVerb(HttpVerb verb);

}

#endif
