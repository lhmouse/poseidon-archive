// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "verbs.hpp"
#include "../cxx_ver.hpp"
#include "../cxx_util.hpp"

namespace Poseidon {
namespace Http {

namespace {
	CONSTEXPR const char g_verb_table[][16] = {
		"INVALID-VERB",
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
	const std::size_t len = std::strlen(str);
	if(len == 0){
		return verb_invalid_verb;
	}
	const char *const begin = g_verb_table[0];
	const AUTO(pos, static_cast<const char *>(::memmem(begin, sizeof(g_verb_table), str, len + 1)));
	if(!pos){
		return verb_invalid_verb;
	}
	std::size_t index = static_cast<std::size_t>(pos - begin) / sizeof(g_verb_table[0]);
	if(pos != g_verb_table[index]){
		return verb_invalid_verb;
	}
	return static_cast<Verb>(index);
}
const char *get_string_from_verb(Verb verb){
	std::size_t index = static_cast<std::size_t>(verb);
	if(index >= COUNT_OF(g_verb_table)){
		index = static_cast<unsigned>(verb_invalid_verb);
	}
	return g_verb_table[index];
}

}
}
