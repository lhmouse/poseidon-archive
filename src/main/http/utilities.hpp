#ifndef POSEIDON_HTTP_UTILITIES_HPP_
#define POSEIDON_HTTP_UTILITIES_HPP_

#include <string>
#include "../optional_map.hpp"

namespace Poseidon {

std::string urlEncode(const std::string &decoded);
std::string urlDecode(const std::string &encoded);

std::string urlEncodedFromOptionalMap(const OptionalMap &decoded);
OptionalMap optionalMapFromUrlEncoded(const std::string &encoded);

}

#endif
