// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

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

	void url_encode_step(std::ostream &os, const std::string &str){
		for(AUTO(it, str.begin()); it != str.end(); ++it){
			const char ch = *it;
			if(ch == ' '){
				os <<'+';
			} else if(is_char_unsafe(ch)){
				os <<'%' <<to_hex_digit(ch >> 4) <<to_hex_digit(ch);
			} else {
				os <<ch;
			}
		}
	}
	char url_decode_step(std::istream &is, std::string &str, const char *stops_at){
		typedef std::istream::traits_type traits;
		traits::int_type next = is.peek();
		const char *pos = "";
		int high = 1, low = 1;
		enum {
			S_PLAIN,
			S_WANTS_HIGH,
			S_WANTS_LOW,
		} state = S_PLAIN;
		for(; !traits::eq_int_type(next, traits::eof()); next = is.peek()){
			const char ch = is.get();
			switch(state){
			case S_PLAIN:
				if(ch == '%'){
					state = S_WANTS_HIGH;
					break;
				}
				pos = std::strchr(stops_at, ch);
				if(pos){
					return *pos;
				}
				if(ch == '+'){
					str += ' ';
				} else {
					str += ch;
				}
				// state = S_PLAIN;
				break;

			case S_WANTS_HIGH:
				high = from_hex_digit(ch);
				if(high < 0){
					is.setstate(std::ios::failbit);
					return 0;
				}
				state = S_WANTS_LOW;
				break;

			case S_WANTS_LOW:
				low = from_hex_digit(ch);
				if(low < 0){
					is.setstate(std::ios::failbit);
					return 0;
				}
				str += static_cast<char>((high << 4) | low);
				state = S_PLAIN;
				break;
			}
		}
		if(state != S_PLAIN){
			is.setstate(std::ios::failbit);
			return 0;
		}
		return 0;
	}
}

namespace Http {
	void url_encode(std::ostream &os, const std::string &str){
		PROFILE_ME;

		url_encode_step(os, str);
	}
	void url_decode(std::istream &is, std::string &str){
		PROFILE_ME;

		str.clear();
		url_decode_step(is, str, "");
	}

	void url_encode_params(std::ostream &os, const OptionalMap &params){
		PROFILE_ME;

		AUTO(it, params.begin());
		if(it != params.end()){
			os <<it->first;
			os <<'=';
			url_encode_step(os, it->second);

			while(++it != params.end()){
				os <<'&';

				os <<it->first;
				os <<'=';
				url_encode_step(os, it->second);
			}
		}
	}
	void url_decode_params(std::istream &is, OptionalMap &params){
		PROFILE_ME;

		params.clear();
		for(;;){
			std::string key, val;
			const char key_term = url_decode_step(is, key, "=&");
			if(!is){
				break;
			}
			if(key_term == '='){
				url_decode_step(is, val, "&");
				if(!is){
					break;
				}
			}
			params.append(SharedNts(key), STD_MOVE(val));
			if(is.eof()){
				break;
			}
		}
	}
}

}
