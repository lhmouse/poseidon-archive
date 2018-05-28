// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_VERBS_HPP_
#define POSEIDON_HTTP_VERBS_HPP_

namespace Poseidon {
namespace Http {

using Verb = int;

inline namespace Verbs {
	enum {
		verb_invalid  = -1,
		verb_get      =  0,
		verb_post     =  1,
		verb_head     =  2,
		verb_put      =  3,
		verb_delete   =  4,
		verb_trace    =  5,
		verb_connect  =  6,
		verb_options  =  7,
	};
}

extern Verb get_verb_from_string(const char *str);
extern const char * get_string_from_verb(Verb verb);

}
}

#endif
