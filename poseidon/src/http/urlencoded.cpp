// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "urlencoded.hpp"
#include "../profiler.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {
namespace Http {

namespace {
	CONSTEXPR const bool g_unsafe_char_table[256] = {
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
	CONSTEXPR const char g_hex_table[32] = {
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
	};

	bool is_char_unsafe(char ch){
		return g_unsafe_char_table[ch & 0xFF];
	}

	char to_hex_digit(int byte, bool upper_case = false){
		return g_hex_table[(byte & 0x0F) + upper_case * sizeof(g_hex_table) / 2];
	}
	int from_hex_digit(char ch){
		const AUTO(p, static_cast<const char *>(std::memchr(g_hex_table, ch, sizeof(g_hex_table))));
		if(!p){
			return -1;
		}
		return (p - g_hex_table) & 0x0F;
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
			state_plain,
			state_wants_high,
			state_wants_low,
		} state = state_plain;
		for(; !traits::eq_int_type(next, traits::eof()); next = is.peek()){
			const char ch = traits::to_char_type(is.get());
			switch(state){
			case state_plain:
				if(ch == '%'){
					state = state_wants_high;
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
				// state = state_plain;
				break;

			case state_wants_high:
				high = from_hex_digit(ch);
				if(high < 0){
					is.setstate(std::ios::failbit);
					return 0;
				}
				state = state_wants_low;
				break;

			case state_wants_low:
				low = from_hex_digit(ch);
				if(low < 0){
					is.setstate(std::ios::failbit);
					return 0;
				}
				str += static_cast<char>((high << 4) | low);
				state = state_plain;
				break;
			}
		}
		if(state != state_plain){
			is.setstate(std::ios::failbit);
			return 0;
		}
		return 0;
	}
}

void url_encode(std::ostream &os, const std::string &str){
	POSEIDON_PROFILE_ME;

	url_encode_step(os, str);
}
void url_decode(std::istream &is, std::string &str){
	POSEIDON_PROFILE_ME;

	str.clear();
	url_decode_step(is, str, "");
}

void url_encode_params(std::ostream &os, const Option_map &params){
	POSEIDON_PROFILE_ME;

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
void url_decode_params(std::istream &is, Option_map &params){
	POSEIDON_PROFILE_ME;

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
		params.append(Rcnts(key), STD_MOVE(val));
		if(is.eof()){
			break;
		}
	}
}

}
}
