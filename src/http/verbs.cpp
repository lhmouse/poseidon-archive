// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "verbs.hpp"
#include <string.h>

namespace Poseidon {

namespace Http {
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

	Verb get_verb_from_string(const char *str){
		const unsigned len = ::strlen(str);
		if(len == 0){
			return V_INVALID_VERB;
		}
		const char *const begin = VERB_TABLE[0];
		const AUTO(pos, static_cast<const char *>(::memmem(begin, sizeof(VERB_TABLE), str, len + 1)));
		if(!pos){
			return V_INVALID_VERB;
		}
		const unsigned i = (unsigned)(pos - begin) / sizeof(VERB_TABLE[0]);
		if(pos != VERB_TABLE[i]){
			return V_INVALID_VERB;
		}
		return static_cast<Verb>(i);
	}
	const char *get_string_from_verb(Verb verb){
		unsigned i = static_cast<unsigned>(verb);
		if(i >= COUNT_OF(VERB_TABLE)){
			i = static_cast<unsigned>(V_INVALID_VERB);
		}
		return VERB_TABLE[i];
	}
}

}
