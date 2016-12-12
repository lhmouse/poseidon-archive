// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "hex.hpp"
#include "profiler.hpp"

namespace Poseidon {

namespace {
	CONSTEXPR const unsigned char HEX_TABLE[32] = {
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
	};

	unsigned char to_hex_digit(unsigned byte, bool upper_case){
		return HEX_TABLE[(byte & 0x0F) + upper_case * sizeof(HEX_TABLE) / 2];
	}
	int from_hex_digit(unsigned char ch){
		const AUTO(p, static_cast<const unsigned char *>(std::memchr(HEX_TABLE, ch, sizeof(HEX_TABLE))));
		if(!p){
			return -1;
		}
		return (p - HEX_TABLE) & 0x0F;
	}
}

HexEncoder::HexEncoder(bool upper_case)
	: m_upper_case(upper_case)
{
}
HexEncoder::~HexEncoder(){
}

void HexEncoder::clear(){
	m_buffer.clear();
}
void HexEncoder::put(const void *data, std::size_t size){
	PROFILE_ME;

	for(std::size_t i = 0; i < size; ++i){
		const unsigned ch = static_cast<const unsigned char *>(data)[i];
		m_buffer.put(to_hex_digit(ch >> 4, m_upper_case));
		m_buffer.put(to_hex_digit(ch     , m_upper_case));
	}
}
void HexEncoder::put(const StreamBuffer &buffer){
	PROFILE_ME;

	for(AUTO(en, buffer.get_chunk_enumerator()); en; ++en){
		put(en.data(), en.size());
	}
}
StreamBuffer HexEncoder::finalize(){
	PROFILE_ME;

	AUTO(ret, STD_MOVE_IDN(m_buffer));
	clear();
	return ret;
}

HexDecoder::HexDecoder()
	: m_seq(1)
{
}
HexDecoder::~HexDecoder(){
}

void HexDecoder::clear(){
	m_seq = 1;
	m_buffer.clear();
}
void HexDecoder::put(const void *data, std::size_t size){
	PROFILE_ME;

	for(std::size_t i = 0; i < size; ++i){
		const unsigned ch = static_cast<const unsigned char *>(data)[i];
		const int digit = from_hex_digit(ch);
		if(digit < 0){
			continue;
		}
		const unsigned seq = (m_seq << 4) | static_cast<unsigned>(digit);
		if(seq >= 0x0100){
			m_buffer.put(seq);
			m_seq = 1;
		} else {
			m_seq = seq;
		}
	}
}
void HexDecoder::put(const StreamBuffer &buffer){
	PROFILE_ME;

	for(AUTO(en, buffer.get_chunk_enumerator()); en; ++en){
		put(en.data(), en.size());
	}
}
StreamBuffer HexDecoder::finalize(){
	PROFILE_ME;

	AUTO(ret, STD_MOVE_IDN(m_buffer));
	clear();
	return ret;
}

}
