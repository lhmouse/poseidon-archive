#ifndef POSEIDON_PROTOCOL_BASE_HPP_
#define POSEIDON_PROTOCOL_BASE_HPP_

#include <string>
#include <vector>
#include "stream_buffer.hpp"

namespace Poseidon {

}

#define PROTOCOL(name_, id_, ...)	\
	struct name_ {	\
		enum {	\
			PROTOCOL_ID = id_	\
		};	\
		__VA_ARGS__	\
	};

// Protocol Fields
#define PF_VINT50(name_)	\
	long long name_;	\

#define PF_VUINT50(name_)	\
	unsigned long long name_;	\

#define PF_STRING(name_)	\
	::std::string name_;	\

#define PF_REPEAT(name_, ...)	\
	struct UNIQUE_ID {	\
		__VA_ARGS__	\
	};	\
	::std::vector<UNIQUE_ID> name_;	\

#endif
