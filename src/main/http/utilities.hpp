#ifndef POSEIDON_HTTP_UTILITIES_HPP_
#define POSEIDON_HTTP_UTILITIES_HPP_

#include <string>
#include <cstddef>
#include "../optional_map.hpp"

namespace Poseidon {

std::string urlEncode(const void *data, std::size_t size);
std::string urlDecode(const void *data, std::size_t size);

inline std::string urlEncode(const std::string &data){
	return urlEncode(data.data(), data.size());
}
inline std::string urlDecode(const std::string &data){
	return urlDecode(data.data(), data.size());
}

std::string urlEncodedFromOptionalMap(const OptionalMap &data);
OptionalMap optionalMapFromUrlEncoded(const std::string &data);

std::string hexEncode(const void *data, std::size_t size, bool upperCase = false);
std::string hexDecode(const void *data, std::size_t size);

inline std::string hexEncode(const std::string &data, bool upperCase = false){
	return hexEncode(data.data(), data.size(), upperCase);
}
inline std::string hexDecode(const std::string &data){
	return hexDecode(data.data(), data.size());
}

std::string base64Encode(const void *data, std::size_t size);
std::string base64Decode(const void *data, std::size_t size);

inline std::string base64Encode(const std::string &data){
	return base64Encode(data.data(), data.size());
}
inline std::string base64Decode(const std::string &data){
	return base64Decode(data.data(), data.size());
}

}

#endif
