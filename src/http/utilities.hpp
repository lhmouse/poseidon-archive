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
	extern std::string url_encode(const void *data, std::size_t size);
	extern std::string url_decode(const void *data, std::size_t size);

	inline std::string url_encode(const char *data){
		return url_encode(data, std::strlen(data));
	}
	inline std::string url_decode(const char *data){
		return url_decode(data, std::strlen(data));
	}

	inline std::string url_encode(const std::string &data){
		return url_encode(data.data(), data.size());
	}
	inline std::string url_decode(const std::string &data){
		return url_decode(data.data(), data.size());
	}

	extern std::string url_encoded_from_optional_map(const OptionalMap &data);
	extern OptionalMap optional_map_from_url_encoded(const std::string &data);

	extern std::string hex_encode(const void *data, std::size_t size, bool upper_case = false);
	extern std::string hex_decode(const void *data, std::size_t size);

	inline std::string hex_encode(const char *data, bool upper_case = false){
		return hex_encode(data, std::strlen(data), upper_case);
	}
	inline std::string hex_decode(const char *data){
		return hex_decode(data, std::strlen(data));
	}

	inline std::string hex_encode(const std::string &data, bool upper_case = false){
		return hex_encode(data.data(), data.size(), upper_case);
	}
	inline std::string hex_decode(const std::string &data){
		return hex_decode(data.data(), data.size());
	}

	extern std::string base64_encode(const void *data, std::size_t size);
	extern std::string base64_decode(const void *data, std::size_t size);

	inline std::string base64_encode(const char *data){
		return base64_encode(data, std::strlen(data));
	}
	inline std::string base64_decode(const char *data){
		return base64_decode(data, std::strlen(data));
	}

	inline std::string base64_encode(const std::string &data){
		return base64_encode(data.data(), data.size());
	}
	inline std::string base64_decode(const std::string &data){
		return base64_decode(data.data(), data.size());
	}
}

}

#endif
