// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_VERBS_HPP_
#define POSEIDON_HTTP_VERBS_HPP_

namespace Poseidon {

namespace Http {
	typedef unsigned Verb;

	namespace Verbs {
		enum {
			V_INVALID_VERB	= 0,
			V_GET			= 1,
			V_POST			= 2,
			V_HEAD			= 3,
			V_PUT			= 4,
			V_DELETE		= 5,
			V_TRACE			= 6,
			V_CONNECT		= 7,
			V_OPTIONS		= 8,
		};
	}

	using namespace Verbs;

	extern Verb get_verb_from_string(const char *str);
	extern const char *get_string_from_verb(Verb verb);
}

}

#endif
