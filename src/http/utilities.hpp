// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_UTILITIES_HPP_
#define POSEIDON_HTTP_UTILITIES_HPP_

#include <string>
#include <cstring>
#include <cstddef>
#include "../optional_map.hpp"

namespace Poseidon {

namespace Http {
	extern std::string urlEncode(const void *data, std::size_t size);
	extern std::string urlDecode(const void *data, std::size_t size);

	inline std::string urlEncode(const char *data){
		return urlEncode(data, std::strlen(data));
	}
	inline std::string urlDecode(const char *data){
		return urlDecode(data, std::strlen(data));
	}

	inline std::string urlEncode(const std::string &data){
		return urlEncode(data.data(), data.size());
	}
	inline std::string urlDecode(const std::string &data){
		return urlDecode(data.data(), data.size());
	}

	extern std::string urlEncodedFromOptionalMap(const OptionalMap &data);
	extern OptionalMap optionalMapFromUrlEncoded(const std::string &data);

	extern std::string hexEncode(const void *data, std::size_t size, bool upperCase = false);
	extern std::string hexDecode(const void *data, std::size_t size);

	inline std::string hexEncode(const char *data, bool upperCase = false){
		return hexEncode(data, std::strlen(data), upperCase);
	}
	inline std::string hexDecode(const char *data){
		return hexDecode(data, std::strlen(data));
	}

	inline std::string hexEncode(const std::string &data, bool upperCase = false){
		return hexEncode(data.data(), data.size(), upperCase);
	}
	inline std::string hexDecode(const std::string &data){
		return hexDecode(data.data(), data.size());
	}

	extern std::string base64Encode(const void *data, std::size_t size);
	extern std::string base64Decode(const void *data, std::size_t size);

	inline std::string base64Encode(const char *data){
		return base64Encode(data, std::strlen(data));
	}
	inline std::string base64Decode(const char *data){
		return base64Decode(data, std::strlen(data));
	}

	inline std::string base64Encode(const std::string &data){
		return base64Encode(data.data(), data.size());
	}
	inline std::string base64Decode(const std::string &data){
		return base64Decode(data.data(), data.size());
	}
}

}

#endif
