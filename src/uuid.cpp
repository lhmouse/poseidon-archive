// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "uuid.hpp"
#include <sys/types.h>
#include <unistd.h>
#include "atomic.hpp"
#include "endian.hpp"
#include "exception.hpp"
#include "time.hpp"
#include "random.hpp"

namespace Poseidon {

namespace {
	CONSTEXPR const unsigned char BYTES_MIN[16] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
	CONSTEXPR const unsigned char BYTES_MAX[16] = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };

	const Uuid g_min_uuid(BYTES_MIN);
	const Uuid g_max_uuid(BYTES_MAX);

	const unsigned g_pid = static_cast<boost::uint16_t>(::getpid());
	volatile boost::uint32_t g_auto_inc = 0;
}

const Uuid &Uuid::min() NOEXCEPT {
	return g_min_uuid;
}
const Uuid &Uuid::max() NOEXCEPT {
	return g_max_uuid;
}

Uuid Uuid::random() NOEXCEPT {
	const AUTO(utc_now, get_utc_time());
	const AUTO(unique, (atomic_add(g_auto_inc, 1, ATOMIC_RELAXED) << 16) | g_pid);
	union {
		unsigned char bytes[16];
		boost::uint16_t u16[8];
		boost::uint32_t u32[4];
	} un;
	store_be(un.u32[0], utc_now >> 12);
	store_be(un.u16[2], (utc_now << 4) | ((unique >> 26) & 0x000F));
	store_be(un.u16[3], 0xE000 | ((unique >> 14) & 0x0FFFu)); // 版本 = 14
	store_be(un.u16[4], 0xC000 | (unique & 0x3FFF)); // 变种 = 3
	store_be(un.u16[5], random_uint32());
	store_be(un.u32[3], random_uint32());
	return Uuid(un.bytes);
}

Uuid::Uuid(const char (&str)[36]){
	DEBUG_THROW_UNLESS(from_string(str), Exception, sslit("Invalid UUID string"));
}
Uuid::Uuid(const std::string &str){
	DEBUG_THROW_UNLESS(from_string(str), Exception, sslit("Invalid UUID string"));
}

void Uuid::to_string(char (&str)[36], bool upper_case) const {
	AUTO(read, begin());
	char *write = str;

#define PRINT(count_)	\
	for(std::size_t i = 0; i < count_; ++i){	\
		const unsigned byte = *(read++);	\
		unsigned ch = byte >> 4;	\
		if(ch <= 9){	\
			ch += '0';	\
		} else if(upper_case){	\
			ch += 'A' - 0x0A;	\
		} else {	\
			ch += 'a' - 0x0A;	\
		}	\
		*(write++) = ch;	\
		ch = byte & 0x0F;	\
		if(ch <= 9){	\
			ch += '0';	\
		} else if(upper_case){	\
			ch += 'A' - 0x0A;	\
		} else {	\
			ch += 'a' - 0x0A;	\
		}	\
		*(write++) = ch;	\
	}

	PRINT(4) *(write++) = '-';
	PRINT(2) *(write++) = '-';
	PRINT(2) *(write++) = '-';
	PRINT(2) *(write++) = '-';
	PRINT(6)
}
void Uuid::to_string(std::string &str, bool upper_case) const {
	str.resize(36);
	to_string(reinterpret_cast<char (&)[36]>(str[0]), upper_case);
}
std::string Uuid::to_string(bool upper_case) const {
	std::string str;
	to_string(str, upper_case);
	return str;
}
bool Uuid::from_string(const char (&str)[36]){
	const char *read = str;
	AUTO(write, begin());

#define SCAN(count_)	\
	for(std::size_t i = 0; i < count_; ++i){	\
		unsigned byte;	\
		unsigned ch = (unsigned char)*(read++);	\
		if(('0' <= ch) && (ch <= '9')){	\
			ch -= '0';	\
		} else if(('A' <= ch) && (ch <= 'F')){	\
			ch -= 'A' - 0x0A;	\
		} else if(('a' <= ch) && (ch <= 'f')){	\
			ch -= 'a' - 0x0A;	\
		} else {	\
			return false;	\
		}	\
		byte = ch << 4;	\
		ch = (unsigned char)*(read++);	\
		if(('0' <= ch) && (ch <= '9')){	\
			ch -= '0';	\
		} else if(('A' <= ch) && (ch <= 'F')){	\
			ch -= 'A' - 0x0A;	\
		} else if(('a' <= ch) && (ch <= 'f')){	\
			ch -= 'a' - 0x0A;	\
		} else {	\
			return false;	\
		}	\
		byte |= ch;	\
		*(write++) = byte;	\
	}

	SCAN(4) if(*(read++) != '-'){ return false; }
	SCAN(2) if(*(read++) != '-'){ return false; }
	SCAN(2) if(*(read++) != '-'){ return false; }
	SCAN(2) if(*(read++) != '-'){ return false; }
	SCAN(6)

	return true;
}
bool Uuid::from_string(const std::string &str){
	if(str.size() != 36){
		return false;
	}
	return from_string(reinterpret_cast<const char (&)[36]>(str[0]));
}

std::ostream &operator<<(std::ostream &os, const Uuid &rhs){
	char temp[37];
	rhs.to_string(reinterpret_cast<char (&)[36]>(temp));
	temp[36] = 0;
	return os <<temp;
}

}
