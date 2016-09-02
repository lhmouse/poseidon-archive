// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "oid.hpp"
#pragma GCC push_options
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <bson.h>
#pragma GCC pop_options
#include "../exception.hpp"

namespace Poseidon {

namespace MongoDb {
	Oid Oid::random() NOEXCEPT {
		::bson_oid_t oid;
		::bson_oid_init(&oid, NULLPTR);

		return Oid(oid.bytes);
	}
	Oid Oid::min() NOEXCEPT {
		static CONSTEXPR const unsigned char bytes[12] = {
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

		return Oid(bytes);
	}
	Oid Oid::max() NOEXCEPT {
		static CONSTEXPR const unsigned char bytes[12] = {
			0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };

		return Oid(bytes);
	}

	Oid::Oid(const char (&str)[24]){
		if(!from_string(str)){
			DEBUG_THROW(Exception, sslit("Invalid BSON ObjectId string"));
		}
	}
	Oid::Oid(const std::string &str){
		if(!from_string(str)){
			DEBUG_THROW(Exception, sslit("Invalid BSON ObjectId string"));
		}
	}

	void Oid::to_string(char (&str)[24], bool upper_case) const {
		AUTO(read, begin());
		char *write = str;

		for(std::size_t i = 0; i < 24; ++i){
			const unsigned byte = *(read++);
			unsigned ch = byte >> 4;
			if(ch <= 9){
				ch += '0';
			} else if(upper_case){
				ch += 'A' - 0x0A;
			} else {
				ch += 'a' - 0x0A;
			}
			*(write++) = ch;
			ch = byte & 0x0F;
			if(ch <= 9){
				ch += '0';
			} else if(upper_case){
				ch += 'A' - 0x0A;
			} else {
				ch += 'a' - 0x0A;
			}
			*(write++) = ch;
		}
	}
	void Oid::to_string(std::string &str, bool upper_case) const {
		str.resize(24);
		to_string(reinterpret_cast<char (&)[24]>(str[0]), upper_case);
	}
	std::string Oid::to_string(bool upper_case) const {
		std::string str;
		to_string(str, upper_case);
		return str;
	}
	bool Oid::from_string(const char (&str)[24]){
		const char *read = str;
		AUTO(write, begin());

		for(std::size_t i = 0; i < 24; ++i){
			unsigned byte;
			unsigned ch = (unsigned char)*(read++);
			if(('0' <= ch) && (ch <= '9')){
				ch -= '0';
			} else if(('A' <= ch) && (ch <= 'F')){
				ch -= 'A' - 0x0A;
			} else if(('a' <= ch) && (ch <= 'f')){
				ch -= 'a' - 0x0A;
			} else {
				return false;
			}
			byte = ch << 4;
			ch = (unsigned char)*(read++);
			if(('0' <= ch) && (ch <= '9')){
				ch -= '0';
			} else if(('A' <= ch) && (ch <= 'F')){
				ch -= 'A' - 0x0A;
			} else if(('a' <= ch) && (ch <= 'f')){
				ch -= 'a' - 0x0A;
			} else {
				return false;
			}
			byte |= ch;
			*(write++) = byte;
		}
		return true;
	}
	bool Oid::from_string(const std::string &str){
		if(str.size() != 24){
			return false;
		}
		return from_string(reinterpret_cast<const char (&)[24]>(str[0]));
	}

	std::ostream &operator<<(std::ostream &os, const Oid &rhs){
		char temp[24];
		rhs.to_string(temp);
		return os.write(temp, 24);
	}
}

}
