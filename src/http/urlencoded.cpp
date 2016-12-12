// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "urlencoded.hpp"
#include "../profiler.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {

namespace {
	CONSTEXPR const bool UNSAFE_CHAR_TABLE[256] = {
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	};

	bool is_char_unsafe(char ch){
		return UNSAFE_CHAR_TABLE[ch & 0xFF];
	}

	CONSTEXPR const char HEX_TABLE[32] = {
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
	};

	char to_hex_digit(int byte, bool upper_case = false){
		return HEX_TABLE[(byte & 0x0F) + upper_case * sizeof(HEX_TABLE) / 2];
	}
	int from_hex_digit(char ch){
		const AUTO(p, static_cast<const char *>(std::memchr(HEX_TABLE, ch, sizeof(HEX_TABLE))));
		if(!p){
			return -1;
		}
		return (p - HEX_TABLE) & 0x0F;
	}
}

namespace Http {
	void url_encode(std::ostream &os, const std::string &str){
		PROFILE_ME;

		for(AUTO(it, str.begin()); it != str.end(); ++it){
			const int ch = *it;
			if(ch == ' '){
				os <<'+';
			} else if(is_char_unsafe(ch)){
				os <<'%' <<to_hex_digit(ch >> 4) <<to_hex_digit(ch);
			} else {
				os <<ch;
			}
		}
	}
	void url_decode(std::istream &is, std::string &str){
		PROFILE_ME;

		str.clear();

		char ch;
		while(is.get(ch)){
			if(ch == '+'){
				str += ' ';
			} else if(ch == '%'){
				int high, low;
				if(!is.get(ch) || ((high = from_hex_digit(ch)) < 0)){
					is.setstate(std::ios::badbit);
					break;
				}
				if(!is.get(ch) || ((low = from_hex_digit(ch)) < 0)){
					is.setstate(std::ios::badbit);
					break;
				}
				str += (char)((high << 4) | low);
			} else {
				str += ch;
			}
		}
	}

	void url_encode_params(std::ostream &os, const OptionalMap &params){
		PROFILE_ME;

		AUTO(it, params.begin());
		if(it != params.end()){
			os <<it->first;
			os <<'=';
			url_encode(os, it->second);

			while(++it != params.end()){
				os <<'&';

				os <<it->first;
				os <<'=';
				url_encode(os, it->second);
			}
		}
	}
	void url_decode_params(std::istream &is, OptionalMap &params){
		PROFILE_ME;

		params.clear();

		std::string line;
		while(std::getline(is, line, '&')){
			SharedNts key;
			std::string value;
			std::size_t equ = line.find('=');
			if(equ == std::string::npos){
				key = SharedNts(line.data(), line.size());
				// value = std::string;
			} else {
				key = SharedNts(line.data(), equ);
				StreamBuffer value_data(line.data() + equ + 1, line.size() - equ - 1);
				Buffer_istream value_is(STD_MOVE(value_data));
				url_decode(value_is, value);
			}
			params.set(STD_MOVE(key), STD_MOVE(value));
		}
	}
}

}
