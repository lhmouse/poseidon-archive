// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_VERBS_HPP_
#define POSEIDON_HTTP_VERBS_HPP_

namespace Poseidon {
namespace Http {

typedef unsigned Verb;

namespace Verbs {
	enum {
		verb_invalid_verb  = 0,
		verb_get           = 1,
		verb_post          = 2,
		verb_head          = 3,
		verb_put           = 4,
		verb_delete        = 5,
		verb_trace         = 6,
		verb_connect       = 7,
		verb_options       = 8,
	};
}

using namespace Verbs;

extern Verb get_verb_from_string(const char *str);
extern const char * get_string_from_verb(Verb verb);

}
}

#endif
