// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "hex.hpp"
#include "profiler.hpp"
#include "exception.hpp"

namespace Poseidon {

namespace {
	CONSTEXPR const unsigned char g_hex_table[32] = {
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
	};

	unsigned char to_hex_digit(unsigned byte, bool upper_case){
		return g_hex_table[(byte & 0x0F) + upper_case * sizeof(g_hex_table) / 2];
	}
	int from_hex_digit(unsigned char ch){
		const AUTO(p, static_cast<const unsigned char *>(std::memchr(g_hex_table, ch, sizeof(g_hex_table))));
		if(!p){
			return -1;
		}
		return (p - g_hex_table) & 0x0F;
	}
}

Hex_encoder::Hex_encoder(bool upper_case)
	: m_upper_case(upper_case)
{
	//
}
Hex_encoder::~Hex_encoder(){
	//
}

void Hex_encoder::clear(){
	m_buffer.clear();
}
void Hex_encoder::put(const void *data, std::size_t size){
	PROFILE_ME;

	for(std::size_t i = 0; i < size; ++i){
		const unsigned ch = static_cast<const unsigned char *>(data)[i];
		m_buffer.put(to_hex_digit(ch >> 4, m_upper_case));
		m_buffer.put(to_hex_digit(ch >> 0, m_upper_case));
	}
}
void Hex_encoder::put(const Stream_buffer &buffer){
	PROFILE_ME;

	const void *data;
	std::size_t size;
	Stream_buffer::Enumeration_cookie cookie;
	while(buffer.enumerate_chunk(&data, &size, cookie)){
		put(data, size);
	}
}
Stream_buffer Hex_encoder::finalize(){
	PROFILE_ME;

	AUTO(ret, STD_MOVE_IDN(m_buffer));
	clear();
	return ret;
}

Hex_decoder::Hex_decoder()
	: m_seq(1)
{
	//
}
Hex_decoder::~Hex_decoder(){
	//
}

void Hex_decoder::clear(){
	m_seq = 1;
	m_buffer.clear();
}
void Hex_decoder::put(const void *data, std::size_t size){
	PROFILE_ME;

	for(std::size_t i = 0; i < size; ++i){
		const unsigned char ch = static_cast<const unsigned char *>(data)[i];
		if((ch == ' ') || (ch == '\t') || (ch == '\r') || (ch == '\n')){
			continue;
		}
		const int digit = from_hex_digit(ch);
		DEBUG_THROW_UNLESS(digit >= 0, Exception, Rcnts::view("Invalid hex character encountered"));
		unsigned seq = m_seq << 4;
		seq += static_cast<unsigned>(digit);
		if(seq >= 0x0100){
			m_buffer.put(seq & 0xFF);
			m_seq = 1;
		} else {
			m_seq = seq;
		}
	}
}
void Hex_decoder::put(const Stream_buffer &buffer){
	PROFILE_ME;

	const void *data;
	std::size_t size;
	Stream_buffer::Enumeration_cookie cookie;
	while(buffer.enumerate_chunk(&data, &size, cookie)){
		put(data, size);
	}
}
Stream_buffer Hex_decoder::finalize(){
	PROFILE_ME;

	DEBUG_THROW_UNLESS(m_seq == 1, Exception, Rcnts::view("Incomplete hex data"));
	AUTO(ret, STD_MOVE_IDN(m_buffer));
	clear();
	return ret;
}

std::string hex_encode(const void *data, std::size_t size, bool upper_case){
	PROFILE_ME;

	Hex_encoder enc(upper_case);
	enc.put(data, size);
	return enc.get_buffer().dump_string();
}
std::string hex_encode(const char *str, bool upper_case){
	PROFILE_ME;

	Hex_encoder enc(upper_case);
	enc.put(str);
	return enc.get_buffer().dump_string();
}
std::string hex_encode(const std::string &str, bool upper_case){
	PROFILE_ME;

	Hex_encoder enc(upper_case);
	enc.put(str);
	return enc.get_buffer().dump_string();
}

std::string hex_decode(const void *data, std::size_t size){
	PROFILE_ME;

	Hex_decoder dec;
	dec.put(data, size);
	return dec.get_buffer().dump_string();
}
std::string hex_decode(const char *str){
	PROFILE_ME;

	Hex_decoder dec;
	dec.put(str);
	return dec.get_buffer().dump_string();
}
std::string hex_decode(const std::string &str){
	PROFILE_ME;

	Hex_decoder dec;
	dec.put(str);
	return dec.get_buffer().dump_string();
}

}
