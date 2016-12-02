// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "string.hpp"
#include "log.hpp"
#include <iomanip>

namespace Poseidon {

namespace {
	const std::string g_empty_string;
}

const std::string &empty_string() NOEXCEPT {
	return g_empty_string;
}

bool is_valid_utf8_string(const std::string &str){
	boost::uint32_t code_point;
	for(AUTO(it, str.begin()); it != str.end(); ++it){
		code_point = static_cast<unsigned char>(*it);
		if((code_point & 0x80u) == 0){
			continue;
		}
		const AUTO(bytes, (unsigned)__builtin_clz((~code_point | 1) & 0xFF) - (sizeof(unsigned) - 1) * CHAR_BIT);
		if(bytes - 2 > 2){ // 2, 3, 4
			LOG_POSEIDON_WARNING("Invalid UTF-8 leading byte: bytes = ", bytes);
			return false;
		}
		code_point &= (0xFFu >> bytes);
		for(unsigned i = 1; i < bytes; ++i){
			++it;
			if(it == str.end()){
				LOG_POSEIDON_WARNING("String is truncated.");
				return false;
			}
			const unsigned trailing = static_cast<unsigned char>(*it);
			if((trailing & 0xC0u) != 0x80u){
				LOG_POSEIDON_WARNING("Invalid UTF-8 trailing byte: trailing = 0x", std::hex, std::setw(2), std::setfill('0'), trailing);
				return false;
			}
		}
		if(code_point < 0x80u){
			LOG_POSEIDON_WARNING("UTF-8 code point is overlong: code_point = 0x", std::hex, std::setw(4), std::setfill('0'), code_point);
			return false;
		} else if((code_point < 0x800u) && (bytes > 2)){
			LOG_POSEIDON_WARNING("UTF-8 code point is overlong: code_point = 0x", std::hex, std::setw(4), std::setfill('0'), code_point);
			return false;
		} else if((code_point < 0x10000u) && (bytes > 3)){
			LOG_POSEIDON_WARNING("UTF-8 code point is overlong: code_point = 0x", std::hex, std::setw(4), std::setfill('0'), code_point);
			return false;
		} else if(code_point > 0x10FFFFu){
			LOG_POSEIDON_WARNING("Invalid UTF-8 code point: code_point = 0x", std::hex, std::setw(6), std::setfill('0'), code_point);
			return false;
		}
		if(code_point - 0xD800u < 0x800u){
			LOG_POSEIDON_WARNING("UTF-8 code point is reserved: code_point = 0x", std::hex, std::setw(4), std::setfill('0'), code_point);
			return false;
		}
	}
	return true;
}

}
