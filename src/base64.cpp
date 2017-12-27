// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "base64.hpp"
#include "profiler.hpp"
#include "protocol_exception.hpp"

namespace Poseidon {

namespace {
	CONSTEXPR const unsigned char BASE64_TABLE[64] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/',
	};

	unsigned char to_base64_digit(unsigned long byte){
		return BASE64_TABLE[byte & 0x3F];
	}

	CONSTEXPR const signed char BASE64_REV_TABLE[256] = {
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,
		52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1,
		-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
		15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,
		-1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
		41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	};

	int from_base64_digit(unsigned char ch){
		return BASE64_REV_TABLE[ch & 0xFF];
	}
}

Base64Encoder::Base64Encoder()
	: m_seq(1)
{ }
Base64Encoder::~Base64Encoder(){ }

void Base64Encoder::clear(){
	m_seq = 1;
	m_buffer.clear();
}
void Base64Encoder::put(const void *data, std::size_t size){
	PROFILE_ME;

	for(std::size_t i = 0; i < size; ++i){
		const unsigned char ch = static_cast<const unsigned char *>(data)[i];
		unsigned long seq = m_seq << 8;
		seq += static_cast<unsigned>(ch);
		if(seq >= (1ul << 24)){
			m_buffer.put(to_base64_digit(seq >> 18));
			m_buffer.put(to_base64_digit(seq >> 12));
			m_buffer.put(to_base64_digit(seq >>  6));
			m_buffer.put(to_base64_digit(seq >>  0));
			m_seq = 1;
		} else {
			m_seq = seq;
		}
	}
}
void Base64Encoder::put(const StreamBuffer &buffer){
	PROFILE_ME;

	const void *data;
	std::size_t size;
	StreamBuffer::EnumerationCookie cookie;
	while(buffer.enumerate_chunk(&data, &size, cookie)){
		put(data, size);
	}
}
StreamBuffer Base64Encoder::finalize(){
	PROFILE_ME;

	const AUTO(seq, m_seq);
	if(seq >= (1ul << 16)){
		m_buffer.put(to_base64_digit(seq >> 10));
		m_buffer.put(to_base64_digit(seq >>  4));
		m_buffer.put(to_base64_digit(seq <<  2));
		m_buffer.put('=');
	} else if(seq >= (1ul << 8)){
		m_buffer.put(to_base64_digit(seq >>  2));
		m_buffer.put(to_base64_digit(seq <<  4));
		m_buffer.put('=');
		m_buffer.put('=');
	}

	AUTO(ret, STD_MOVE_IDN(m_buffer));
	clear();
	return ret;
}

Base64Decoder::Base64Decoder()
	: m_seq(1)
{ }
Base64Decoder::~Base64Decoder(){ }

void Base64Decoder::clear(){
	m_seq = 1;
	m_buffer.clear();
}
void Base64Decoder::put(const void *data, std::size_t size){
	PROFILE_ME;

	for(std::size_t i = 0; i < size; ++i){
		const unsigned char ch = static_cast<const unsigned char *>(data)[i];
		if((ch == ' ') || (ch == '\t') || (ch == '\r') || (ch == '\n')){
			continue;
		}
		unsigned long seq = m_seq << 6;
		if(ch == '='){
			unsigned long n_add = 0;
			if((seq >= (1ul << 24)) && ((seq >> 24) <= 2)){
				n_add = 1ul << 24;
			} else if((seq >= (1ul << 18)) && ((seq >> 18) <= 1)){
				n_add = 1ul << 18;
			}
			DEBUG_THROW_UNLESS(n_add != 0, ProtocolException, sslit("Invalid base64 padding character encountered"), -1);
			seq += n_add;
		} else {
			const int digit = from_base64_digit(ch);
			DEBUG_THROW_UNLESS(digit >= 0, ProtocolException, sslit("Invalid base64 character encountered"), -1);
			seq += static_cast<unsigned>(digit);
		}
		if(seq >= (1ul << 24)){
			const unsigned n = static_cast<unsigned>(4 - (seq >> 24));
			switch(n){
			case 1:
				m_buffer.put(static_cast<unsigned char>(seq >> 16));
				break;
			case 2:
				m_buffer.put(static_cast<unsigned char>(seq >> 16));
				m_buffer.put(static_cast<unsigned char>(seq >>  8));
				break;
			case 3:
				m_buffer.put(static_cast<unsigned char>(seq >> 16));
				m_buffer.put(static_cast<unsigned char>(seq >>  8));
				m_buffer.put(static_cast<unsigned char>(seq >>  0));
				break;
			default:
				DEBUG_THROW(ProtocolException, sslit("Invalid base64 data"), -1);
			}
			m_seq = 1;
		} else {
			m_seq = seq;
		}
	}
}
void Base64Decoder::put(const StreamBuffer &buffer){
	PROFILE_ME;

	const void *data;
	std::size_t size;
	StreamBuffer::EnumerationCookie cookie;
	while(buffer.enumerate_chunk(&data, &size, cookie)){
		put(data, size);
	}
}
StreamBuffer Base64Decoder::finalize(){
	PROFILE_ME;

	DEBUG_THROW_UNLESS(m_seq == 1, ProtocolException, sslit("Incomplete base64 data"), -1);
	AUTO(ret, STD_MOVE_IDN(m_buffer));
	clear();
	return ret;
}

std::string base64_encode(const void *data, std::size_t size){
	PROFILE_ME;

	Base64Encoder enc;
	enc.put(data, size);
	return enc.get_buffer().dump_string();
}
std::string base64_encode(const char *str){
	PROFILE_ME;

	Base64Encoder enc;
	enc.put(str);
	return enc.get_buffer().dump_string();
}
std::string base64_encode(const std::string &str){
	PROFILE_ME;

	Base64Encoder enc;
	enc.put(str);
	return enc.get_buffer().dump_string();
}

std::string base64_decode(const void *data, std::size_t size){
	PROFILE_ME;

	Base64Decoder dec;
	dec.put(data, size);
	return dec.get_buffer().dump_string();
}
std::string base64_decode(const char *str){
	PROFILE_ME;

	Base64Decoder dec;
	dec.put(str);
	return dec.get_buffer().dump_string();
}
std::string base64_decode(const std::string &str){
	PROFILE_ME;

	Base64Decoder dec;
	dec.put(str);
	return dec.get_buffer().dump_string();
}

}
