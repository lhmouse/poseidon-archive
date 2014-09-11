#include "../../precompiled.hpp"
#include "utilities.hpp"
#include "../utilities.hpp"
using namespace Poseidon;

namespace {

const bool SAFE_CHARS[256] = {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
	0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
	0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

bool isCharSafe(char ch){
	return SAFE_CHARS[(unsigned char)ch];
}

const char HEX_TABLE[16] = {
	'0', '1', '2', '3', '4', '5', '6', '7',
	'8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
};

char getHex(char ch){
	return HEX_TABLE[(unsigned char)ch];
}

const signed char HEX_LITERAL_TABLE[256] = {
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, -1, -1, -1, -1, -1, -1,
	-1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

int getHexLiteral(char ch){
	return HEX_LITERAL_TABLE[(unsigned char)ch];
}

}

namespace Poseidon {

std::string urlEncode(const std::string &decoded){
	std::string ret;
	const std::size_t size = decoded.size();
	ret.reserve(size + (size >> 1));
	std::size_t i = 0;
	while(i < size){
		const char ch = decoded[i];
		++i;
		if(ch == ' '){
			ret.push_back('+');
			continue;
		}
		if(isCharSafe(ch)){
			ret.push_back(ch);
			continue;
		}
		ret.push_back('%');
		ret.push_back(getHex(ch & 0x0F));
		ret.push_back(getHex((ch >> 4) & 0x0F));
	}
	return STD_MOVE(ret);
}
std::string urlDecode(const std::string &encoded){
	std::string ret;
	const std::size_t size = encoded.size();
	ret.reserve(size);
	std::size_t i = 0;
	while(i < size){
		const char ch = encoded[i];
		++i;
		if(ch == '+'){
			ret.push_back(' ');
			continue;
		}
		if((ch != '%') || ((i + 1) >= size)){
			ret.push_back(ch);
			continue;
		}
		const int high = getHexLiteral(encoded[i]);
		const int low = getHexLiteral(encoded[i + 1]);
		if((high == -1) || (low == -1)){
			ret.push_back(ch);
			continue;
		}
		i += 2;
		ret.push_back((high << 4) | low);
	}
	return STD_MOVE(ret);
}

std::string urlEncodedFromOptionalMap(const OptionalMap &decoded){
	std::string ret;
	std::vector<std::string> parts;
	parts.reserve(decoded.size());
	for(AUTO(it, decoded.begin()); it != decoded.end(); ++it){
		std::string tmp(it->first.get());
		tmp += '=';
		tmp += urlEncode(it->second);

		parts.push_back(std::string());
		parts.back().swap(tmp);
	}
	return STD_MOVE(ret);
}
OptionalMap optionalMapFromUrlEncoded(const std::string &encoded){
	OptionalMap ret;
	const AUTO(parts, explode<std::string>('&', encoded));
	for(AUTO(it, parts.begin()); it != parts.end(); ++it){
		const std::size_t pos = it->find('=');
		if(pos == std::string::npos){
			ret.set(urlDecode(*it), std::string());
		} else {
			ret.set(urlDecode(it->substr(0, pos)), urlDecode(it->substr(pos + 1)));
		}
	}
	return STD_MOVE(ret);
}

}
