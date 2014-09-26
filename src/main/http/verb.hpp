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
