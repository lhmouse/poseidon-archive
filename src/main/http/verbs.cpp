// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "verbs.hpp"
#include <string.h>

namespace Poseidon {

namespace {
	const char VERB_TABLE[][16] = {
		"INVALID_VERB",
		"GET",
		"POST",
		"HEAD",
		"PUT",
		"DELETE",
		"TRACE",
		"CONNECT",
		"OPTIONS",
	};
}

HttpVerb httpVerbFromString(const char *str){
	const unsigned len = ::strlen(str);
	if(len == 0){
		return HTTP_INVALID_VERB;
	}
	const char *const begin = VERB_TABLE[0];
	const AUTO(pos, static_cast<const char *>(::memmem(begin, sizeof(VERB_TABLE), str, len + 1)));
	if(!pos){
		return HTTP_INVALID_VERB;
	}
	const unsigned i = (unsigned)(pos - begin) / sizeof(VERB_TABLE[0]);
	if(pos != VERB_TABLE[i]){
		return HTTP_INVALID_VERB;
	}
	return static_cast<HttpVerb>(i);
}
const char *stringFromHttpVerb(HttpVerb verb){
	unsigned i = static_cast<unsigned>(verb);
	if(i >= COUNT_OF(VERB_TABLE)){
		i = static_cast<unsigned>(HTTP_INVALID_VERB);
	}
	return VERB_TABLE[i];
}

}
