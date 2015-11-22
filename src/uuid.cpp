// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

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
	inline boost::uint32_t rdtsc_low(){
		boost::uint32_t ret;
		__asm__ __volatile__("rdtsc \n" : "=a"(ret) : : "edx");
		return ret;
	}

	const unsigned char MIN_BYTES[16] = {
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
	const unsigned char MAX_BYTES[16] = {
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };

	const boost::uint32_t g_pid = static_cast<boost::uint16_t>(::getpid());

	volatile boost::uint32_t g_auto_inc = 0;
}

const Uuid Uuid::UUID_MIN(MIN_BYTES);
const Uuid Uuid::UUID_MAX(MAX_BYTES);

Uuid Uuid::random(){
	const AUTO(utc_now, get_utc_time());
	const AUTO(unique, ((atomic_add(g_auto_inc, 1, ATOMIC_RELAXED) << 16) & 0x3FFFFFFFu) | g_pid);

	Uuid ret;
	store_be(ret.m_storage.u32[0], utc_now >> 12);
	store_be(ret.m_storage.u16[2], (utc_now << 4) | (unique >> 26));
	store_be(ret.m_storage.u16[3], (unique >> 14) & 0x0FFFu); // 版本 = 0
	store_be(ret.m_storage.u16[4], 0xC000u | (unique & 0x3FFFu)); // 变种 = 3
	store_be(ret.m_storage.u16[5], rand32());
	store_be(ret.m_storage.u32[3], rand32());
	return ret;
}

Uuid::Uuid(const char (&str)[36]){
	if(!from_string(str)){
		throw std::invalid_argument("Invalid UUID string");
	}
}
Uuid::Uuid(const std::string &str){
	if(!from_string(str)){
		throw std::invalid_argument("Invalid UUID string");
	}
}

void Uuid::to_string(char (&str)[36], bool upper_case) const {
	AUTO(read, begin());
	char *write = str;

#define PRINT(count_)   \
	for(std::size_t i = 0; i < count_; ++i){    \
		const unsigned byte = *(read++);    \
		unsigned ch = byte >> 4;    \
		if(ch <= 9){    \
			ch += '0';  \
		} else if(upper_case){  \
			ch += 'A' - 0x0A;   \
		} else {    \
			ch += 'a' - 0x0A;   \
		}   \
		*(write++) = ch;    \
		ch = byte & 0x0F;   \
		if(ch <= 9){    \
			ch += '0';  \
		} else if(upper_case){  \
			ch += 'A' - 0x0A;   \
		} else {    \
			ch += 'a' - 0x0A;   \
		}   \
		*(write++) = ch;    \
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

#define SCAN(count_)    \
	for(std::size_t i = 0; i < count_; ++i){    \
		unsigned byte;  \
		unsigned ch = (unsigned char)*(read++); \
		if(('0' <= ch) && (ch <= '9')){ \
			ch -= '0';  \
		} else if(('A' <= ch) && (ch <= 'F')){  \
			ch -= 'A' - 0x0A;   \
		} else if(('a' <= ch) && (ch <= 'f')){  \
			ch -= 'a' - 0x0A;   \
		} else {    \
			return false;   \
		}   \
		byte = ch << 4; \
		ch = (unsigned char)*(read++);  \
		if(('0' <= ch) && (ch <= '9')){ \
			ch -= '0';  \
		} else if(('A' <= ch) && (ch <= 'F')){  \
			ch -= 'A' - 0x0A;   \
		} else if(('a' <= ch) && (ch <= 'f')){  \
			ch -= 'a' - 0x0A;   \
		} else {    \
			return false;   \
		}   \
		byte |= ch; \
		*(write++) = byte;  \
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
	char temp[36];
	rhs.to_string(temp);
	return os.write(temp, 36);
}

}
