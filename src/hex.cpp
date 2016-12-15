// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "hex.hpp"
#include "protocol_exception.hpp"

namespace Poseidon {

namespace {
	CONSTEXPR const unsigned char HEX_TABLE[32] = {
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
	};

	unsigned decode(unsigned char (&dst)[1], const unsigned char (&src)[2]){
		const void *ptr = std::memchr(HEX_TABLE, src[0], sizeof(HEX_TABLE));
		if(!ptr){
			DEBUG_THROW(ProtocolException, sslit("Invalid hex digit"), -1);
		}
		const unsigned high = (static_cast<const unsigned char *>(ptr) - HEX_TABLE) & 0x0F;
		ptr = std::memchr(HEX_TABLE, src[1], sizeof(HEX_TABLE));
		if(!ptr){
			DEBUG_THROW(ProtocolException, sslit("Invalid hex digit"), -1);
		}
		const unsigned low = (static_cast<const unsigned char *>(ptr) - HEX_TABLE) & 0x0F;
		dst[0] = (high << 4) | low;
		return 1;
	}
	void encode(unsigned char (&dst)[2], const unsigned char (&src)[1], bool upper_case){
		dst[0] = HEX_TABLE[upper_case * 16 + ((src[0] >> 4) & 0x0F)];
		dst[1] = HEX_TABLE[upper_case * 16 + ((src[0] >> 0) & 0x0F)];
	}
}

Hex_streambuf::~Hex_streambuf(){
}

int Hex_streambuf::sync(){
	if(gptr()){
		m_buffer.discard(static_cast<unsigned>(gptr() - eback()) * 2);
		setg(NULLPTR, NULLPTR, NULLPTR);
	}
	return std::streambuf::sync();
}

Hex_streambuf::int_type Hex_streambuf::underflow(){
	if(m_which & std::ios_base::in){
		sync();
		unsigned char src[2], dst[1];
		if(m_buffer.peek(src, 2) < 2){
			return traits_type::eof();
		}
		const unsigned n = decode(dst, src);
		if(n == 0){
			return traits_type::eof();
		}
		std::memcpy(m_get_area, dst, n);
		setg(m_get_area, m_get_area, m_get_area + n);
		return traits_type::to_int_type(*gptr());
	} else {
		return traits_type::eof();
	}
}

Hex_streambuf::int_type Hex_streambuf::pbackfail(Hex_streambuf::int_type c){
	if(m_which & std::ios_base::out){
		if(traits_type::eq_int_type(c, traits_type::eof())){
			return traits_type::eof();
		}
		sync();
		unsigned char src[1], dst[2];
		src[0] = c;
		encode(dst, src, m_upper_case);
		m_buffer.unget(dst[1]);
		m_buffer.unget(dst[0]);
		return c;
	} else {
		return traits_type::eof();
	}
}

Hex_streambuf::int_type Hex_streambuf::overflow(Hex_streambuf::int_type c){
	if(m_which & std::ios_base::out){
		if(traits_type::eq_int_type(c, traits_type::eof())){
			return traits_type::not_eof(c);
		}
		sync();
		unsigned char src[1], dst[2];
		src[0] = c;
		encode(dst, src, m_upper_case);
		m_buffer.put(dst, 2);
		return c;
	} else {
		return traits_type::eof();
	}
}

Hex_istream::~Hex_istream(){
}

Hex_ostream::~Hex_ostream(){
}

Hex_stream::~Hex_stream(){
}

}
