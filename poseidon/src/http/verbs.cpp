// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "verbs.hpp"
#include "../cxx_ver.hpp"

namespace Poseidon {
namespace Http {

namespace {
	constexpr char g_verb_table[][8] = {
		"GET",
		"POST",
		"HEAD",
		"PUT",
		"DELETE",
		"TRACE",
		"CONNECT",
		"OPTIONS",
		"???",
	};
}

Verb get_verb_from_string(const char *str){
	const auto begin = std::begin(g_verb_table);
	const auto end = std::end(g_verb_table) - 1;
	const auto ptr = std::find_if(begin, end, [&](const char *cmp){ return std::strcmp(str, cmp) == 0; });
	if(ptr == end){
		return verb_invalid;
	}
	return static_cast<Verb>(ptr - begin);
}
const char * get_string_from_verb(Verb verb){
	const auto begin = std::begin(g_verb_table);
	const auto end = std::end(g_verb_table) - 1;
	if(static_cast<std::size_t>(verb) >= static_cast<std::size_t>(end - begin)){
		return *end;
	}
	const auto ptr = begin + verb;
	return *ptr;
}

}
}
