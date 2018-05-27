// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "base64.hpp"
#include "profiler.hpp"
#include "exception.hpp"

namespace Poseidon {

namespace {
	CONSTEXPR const unsigned char g_base64_table[64] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/',
	};
	CONSTEXPR const signed char g_base64_reverse_table[256] = {
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

	unsigned char to_base64_digit(unsigned long byte){
		return g_base64_table[byte & 0x3F];
	}
	int from_base64_digit(unsigned char ch){
		return g_base64_reverse_table[ch & 0xFF];
	}
}

Base64_encoder::Base64_encoder()
	: m_seq(1)
{
	//
}
Base64_encoder::~Base64_encoder(){
	//
}

void Base64_encoder::clear(){
	m_seq = 1;
	m_buffer.clear();
}
void Base64_encoder::put(const void *data, std::size_t size){
	POSEIDON_PROFILE_ME;

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
void Base64_encoder::put(const Stream_buffer &buffer){
	POSEIDON_PROFILE_ME;

	const void *data;
	std::size_t size;
	Stream_buffer::Enumeration_cookie cookie;
	while(buffer.enumerate_chunk(&data, &size, cookie)){
		put(data, size);
	}
}
Stream_buffer Base64_encoder::finalize(){
	POSEIDON_PROFILE_ME;

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

Base64_decoder::Base64_decoder()
	: m_seq(1)
{
	//
}
Base64_decoder::~Base64_decoder(){
	//
}

void Base64_decoder::clear(){
	m_seq = 1;
	m_buffer.clear();
}
void Base64_decoder::put(const void *data, std::size_t size){
	POSEIDON_PROFILE_ME;

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
			POSEIDON_THROW_UNLESS(n_add != 0, Exception, Rcnts::view("Invalid base64 padding character encountered"));
			seq += n_add;
		} else {
			const int digit = from_base64_digit(ch);
			POSEIDON_THROW_UNLESS(digit >= 0, Exception, Rcnts::view("Invalid base64 character encountered"));
			seq += static_cast<unsigned>(digit);
		}
		if(seq >= (1ul << 24)){
			const unsigned long n = 4 - (seq >> 24);
			switch(n){
			case 1:
				m_buffer.put((seq >> 16) & 0xFF);
				break;
			case 2:
				m_buffer.put((seq >> 16) & 0xFF);
				m_buffer.put((seq >>  8) & 0xFF);
				break;
			case 3:
				m_buffer.put((seq >> 16) & 0xFF);
				m_buffer.put((seq >>  8) & 0xFF);
				m_buffer.put((seq >>  0) & 0xFF);
				break;
			default:
				POSEIDON_THROW(Exception, Rcnts::view("Invalid base64 data"));
			}
			m_seq = 1;
		} else {
			m_seq = seq;
		}
	}
}
void Base64_decoder::put(const Stream_buffer &buffer){
	POSEIDON_PROFILE_ME;

	const void *data;
	std::size_t size;
	Stream_buffer::Enumeration_cookie cookie;
	while(buffer.enumerate_chunk(&data, &size, cookie)){
		put(data, size);
	}
}
Stream_buffer Base64_decoder::finalize(){
	POSEIDON_PROFILE_ME;

	POSEIDON_THROW_UNLESS(m_seq == 1, Exception, Rcnts::view("Incomplete base64 data"));
	AUTO(ret, STD_MOVE_IDN(m_buffer));
	clear();
	return ret;
}

std::string base64_encode(const void *data, std::size_t size){
	POSEIDON_PROFILE_ME;

	Base64_encoder enc;
	enc.put(data, size);
	return enc.get_buffer().dump_string();
}
std::string base64_encode(const char *str){
	POSEIDON_PROFILE_ME;

	Base64_encoder enc;
	enc.put(str);
	return enc.get_buffer().dump_string();
}
std::string base64_encode(const std::string &str){
	POSEIDON_PROFILE_ME;

	Base64_encoder enc;
	enc.put(str);
	return enc.get_buffer().dump_string();
}

std::string base64_decode(const void *data, std::size_t size){
	POSEIDON_PROFILE_ME;

	Base64_decoder dec;
	dec.put(data, size);
	return dec.get_buffer().dump_string();
}
std::string base64_decode(const char *str){
	POSEIDON_PROFILE_ME;

	Base64_decoder dec;
	dec.put(str);
	return dec.get_buffer().dump_string();
}
std::string base64_decode(const std::string &str){
	POSEIDON_PROFILE_ME;

	Base64_decoder dec;
	dec.put(str);
	return dec.get_buffer().dump_string();
}

}
