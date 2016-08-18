// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "oid.hpp"
#include <sys/types.h>
#include <unistd.h>
#include "../atomic.hpp"
#include "../endian.hpp"
#include "../exception.hpp"
#include "../time.hpp"
#include "../random.hpp"

namespace Poseidon {

namespace {
	const boost::uint32_t g_mid = rand32();
	const boost::uint32_t g_pid = static_cast<boost::uint16_t>(::getpid());

	volatile boost::uint32_t g_auto_inc = 0;
}

namespace MongoDb {
	Oid Oid::random(){
		const AUTO(utc_now, get_utc_time());
		const AUTO(auto_inc, atomic_add(g_auto_inc, 1, ATOMIC_RELAXED));

#define COPY_BE(field_, src_)	\
		do {	\
			boost::uint32_t temp_;	\
			store_be(temp_, src_);	\
			std::memcpy(&(field_), &temp_, sizeof(field_));	\
		} while(0)

		Oid ret;
		COPY_BE(ret.m_storage.uts, utc_now / 1000);
		COPY_BE(ret.m_storage.mid, g_mid);
		COPY_BE(ret.m_storage.pid, g_pid);
		COPY_BE(ret.m_storage.inc, auto_inc);
		return ret;
	}


	Oid::Oid(const char (&str)[24]){
		if(!from_string(str)){
			DEBUG_THROW(Exception, sslit("Invalid ObjectID string"));
		}
	}
	Oid::Oid(const std::string &str){
		if(!from_string(str)){
			DEBUG_THROW(Exception, sslit("Invalid ObjectID string"));
		}
	}

	void Oid::to_string(char (&str)[24], bool upper_case) const {
		AUTO(read, begin());
		char *write = str;

		for(std::size_t i = 0; i < 12; ++i){
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

		for(std::size_t i = 0; i < 12; ++i){
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
