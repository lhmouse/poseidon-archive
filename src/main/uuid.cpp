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
	boost::uint32_t rdtscLow(){
		boost::uint32_t ret;
		__asm__ __volatile__("rdtsc \n" : "=a"(ret) : : "edx");
		return ret;
	}

	const unsigned long g_pidHigh = static_cast<unsigned long>(::getpid()) << 16;

	volatile unsigned g_autoInc = rdtscLow();
}

Uuid Uuid::generate(){
	const AUTO(now, getUtcTime());
	const AUTO(unique, g_pidHigh | (atomicAdd(g_autoInc, 1, ATOMIC_RELAXED) & 0xFFFFu));

	Uuid ret
#ifdef POSEIDON_CXX11
		(nullptr)	// 不初始化。
#endif
		;
	storeBe(ret.m_storage.u32[0], now >> 28);
	storeBe(ret.m_storage.u16[2], now >> 12);
	storeBe(ret.m_storage.u16[3], now & 0x0FFFu); // 版本 = 0
	storeBe(ret.m_storage.u32[2], 0xC0000000u | unique); // 变种 = 3
	storeBe(ret.m_storage.u32[3], rand32());
	return ret;
}

Uuid::Uuid(const char (&str)[36]){
	if(!fromString(str)){
		DEBUG_THROW(Exception, SharedNts::observe("Invalid UUID string"));
	}
}
Uuid::Uuid(const std::string &str){
	if(!fromString(str)){
		DEBUG_THROW(Exception, SharedNts::observe("Invalid UUID string"));
	}
}

void Uuid::toString(char (&str)[36], bool upperCase) const {
	AUTO(read, begin());
	char *write = str;

#define PRINT(count_)	\
	for(std::size_t i = 0; i < count_; ++i){	\
		const unsigned byte = *(read++);	\
		unsigned ch = byte >> 4;	\
		if(ch <= 9){	\
			ch += '0';	\
		} else if(upperCase){	\
			ch += 'A' - 0x0A;	\
		} else {	\
			ch += 'a' - 0x0A;	\
		}	\
		*(write++) = ch;	\
		ch = byte & 0x0F;	\
		if(ch <= 9){	\
			ch += '0';	\
		} else if(upperCase){	\
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
void Uuid::toString(std::string &str, bool upperCase) const {
	str.resize(36);
	toString(reinterpret_cast<char (&)[36]>(str[0]), upperCase);
}
std::string Uuid::toString(bool upperCase) const {
	std::string str;
	toString(str, upperCase);
	return str;
}
bool Uuid::fromString(const char (&str)[36]){
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
bool Uuid::fromString(const std::string &str){
	if(str.size() != 36){
		return false;
	}
	return fromString(reinterpret_cast<const char (&)[36]>(str[0]));
}

std::ostream &operator<<(std::ostream &os, const Uuid &rhs){
	char temp[36];
	rhs.toString(temp);
	return os.write(temp, 36);
}

}
