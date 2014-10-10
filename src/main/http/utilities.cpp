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
	return ret;
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
	return ret;
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
	return ret;
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
	return ret;
}

std::string hexEncode(const std::string &decoded){
	std::string ret;
	ret.reserve(decoded.size() * 2);
	for(AUTO(it, decoded.begin()); it != decoded.end(); ++it){
		ret.push_back(getHex(*it & 0x0F));
		ret.push_back(getHex((*it >> 4) & 0x0F));
	}
	return ret;
}
std::string hexDecode(const std::string &encoded){
	std::string ret;
	ret.reserve(encoded.size() / 2);
	int high = -1;
	for(AUTO(it, encoded.begin()); it != encoded.end(); ++it){
		if(high == -1){
			high = getHexLiteral(*it);
			continue;
		}
		const int low = getHexLiteral(*it);
		if(low == -1){
			continue;
		}
		ret.push_back((high << 4) | low);
		high = -1;
	}
	return ret;
}

std::string base64Encode(const std::string &decoded){
	static const char TABLE[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	std::string ret;
	const std::size_t words = decoded.size() / 3;
	const std::size_t whole = words * 3;
	ret.reserve(words * 4 + 4);
	unsigned word;
	for(std::size_t i = 0; i < whole; i += 3){
		word = (unsigned char)decoded[i];
		ret.push_back(TABLE[(word >> 2) & 0x3F]);
		word <<= 8;
		word |= (unsigned char)decoded[i + 1];
		ret.push_back(TABLE[(word >> 4) & 0x3F]);
		word <<= 8;
		word |= (unsigned char)decoded[i + 2];
		ret.push_back(TABLE[(word >> 6) & 0x3F]);
		ret.push_back(TABLE[word & 0x3F]);
	}
	switch(decoded.size() - whole){
	case 1:
		word = (unsigned char)decoded[whole];
		ret.push_back(TABLE[(word >> 2) & 0x3F]);
		ret.push_back(TABLE[(word << 4) & 0x3F]);
		ret.push_back('=');
		ret.push_back('=');
		break;

	case 2:
		word = (unsigned char)decoded[whole];
		ret.push_back(TABLE[(word >> 2) & 0x3F]);
		word <<= 8;
		word |= (unsigned char)decoded[whole + 1];
		ret.push_back(TABLE[(word >> 4) & 0x3F]);
		ret.push_back(TABLE[(word << 2) & 0x3F]);
		ret.push_back('=');
		break;
	}
	return ret;
}
std::string base64Decode(const std::string &encoded){
	static const unsigned char TABLE[] = {
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x3E, 0xFF, 0xFF, 0xFF, 0x3F,
		0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
		0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
		0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
	};

	std::string ret;
	ret.reserve(encoded.size() / 4 * 3 + 3);
	long word = 0;
	unsigned count = 0;
	for(AUTO(it, encoded.begin()); it != encoded.end(); ++it){
		const unsigned char ch = TABLE[(unsigned char)*it];
		if(ch > 0x3F){
			continue;
		}
		word <<= 6;
		word |= ch;
		if(++count % 4 == 0){
			ret.push_back((char)(word >> 16));
			ret.push_back((char)(word >> 8));
			ret.push_back((char)word);
		}
	}
	switch(count % 4){
	case 2:
		ret.push_back((char)(word >> 4));
		break;

	case 3:
		ret.push_back((char)(word >> 10));
		ret.push_back((char)(word >> 2));
		break;
	}
	return ret;
}

}
