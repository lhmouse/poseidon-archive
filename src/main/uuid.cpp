// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "uuid.hpp"
#include "utilities.hpp"
#include "atomic.hpp"
using namespace Poseidon;

namespace {

struct BrokenDownUuid {
	boost::uint32_t first;
	boost::uint16_t second;
	boost::uint16_t third;
	boost::uint64_t bytes;

	void store(Uuid &dst) const {
		AUTO(write, dst.getBytes());

#define DO_STORE(part_)	\
		{	\
			VALUE_TYPE(part_) temp = part_;	\
			unsigned i = sizeof(part_);	\
			do {	\
				--i;	\
				*write = temp >> (i * CHAR_BIT);	\
				++write;	\
			} while(i != 0);	\
		}

		DO_STORE(first)
		DO_STORE(second)
		DO_STORE(third)
		DO_STORE(bytes)
	}
	void load(const Uuid &src){
		AUTO(read, src.getBytes());

#define DO_LOAD(part_)	\
		{	\
			VALUE_TYPE(part_) temp = 0;	\
			unsigned i = sizeof(part_);	\
			do {	\
				--i;	\
				temp |= (unsigned long long)*read << (i * CHAR_BIT);	\
				++read;	\
			} while(i != 0);	\
			part_ = temp;	\
		}

		DO_LOAD(first)
		DO_LOAD(second)
		DO_LOAD(third)
		DO_LOAD(bytes)
	}

	unsigned print(char *text) const {
		const boost::uint16_t fourth = bytes >> 48;
		const boost::uint64_t fifth = (bytes << 16) >> 16;
		return std::sprintf(text, "%08lX-%04X-%04X-%04X-%012llX",
			(unsigned long)first, (unsigned)second, (unsigned)third,
			(unsigned)fourth, (unsigned long long)fifth);
	}
	bool scan(const char *text){
		unsigned long u1;
		unsigned short u2, u3, u4;
		unsigned long long u5;
		char probe;
		if(std::sscanf(text, "%lX-%hX-%hX-%hX-%llX%c",
			&u1, &u2, &u3, &u4, &u5, &probe) != 5)
		{
			return false;
		}
		first = u1;
		second = u2;
		third = u3;
		bytes = ((unsigned long long)u4 << 48) | u5;
		return true;
	}
};

volatile boost::uint32_t g_autoInc = 0;

}

Uuid Uuid::generate(){
	const AUTO(now, getUtcTime());
	BrokenDownUuid temp;
	temp.first = now >> 28;
	temp.second = now >> 12;
	temp.third = now & 0x0FFF; // version = 0
	temp.bytes = 0xC0000000 | atomicAdd(g_autoInc, 1, ATOMIC_RELAXED); // veriant = 3
	temp.bytes <<= 32;
	temp.bytes |= rand32();
	Uuid ret;
	temp.store(ret);
	return ret;
}

std::string Uuid::toString() const {
	BrokenDownUuid temp;
	temp.load(*this);
	char text[64];
	const unsigned len = temp.print(text);
	return std::string(text, len);
}
bool Uuid::fromString(const std::string &str){
	BrokenDownUuid temp;
	if(!temp.scan(str.c_str())){
		return false;
	}
	temp.store(*this);
	return true;
}

namespace Poseidon {

std::ostream &operator<<(std::ostream &os, const Uuid &rhs){
	BrokenDownUuid temp;
	temp.load(rhs);
	char text[64];
	const unsigned len = temp.print(text);
	return os.write(text, len);
}

}
