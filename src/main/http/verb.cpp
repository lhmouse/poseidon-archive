#include "../precompiled.hpp"
#include "verb.hpp"
using namespace Poseidon;

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

namespace Poseidon {

HttpVerb httpVerbFromString(const char *str){
	for(unsigned i = 0; i < COUNT_OF(VERB_TABLE); ++i){
		if(std::strcmp(str, VERB_TABLE[i]) == 0){
			return static_cast<HttpVerb>(i);
		}
	}
	return HTTP_INVALID_VERB;
}
const char *stringFromHttpVerb(HttpVerb verb){
	unsigned i = static_cast<unsigned>(verb);
	if(i >= COUNT_OF(VERB_TABLE)){
		i = static_cast<unsigned>(HTTP_INVALID_VERB);
	}
	return VERB_TABLE[i];
}

}
