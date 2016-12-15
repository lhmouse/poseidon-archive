// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "base64.hpp"
#include "protocol_exception.hpp"

namespace Poseidon {

namespace {
	CONSTEXPR const unsigned char BASE64_TABLE[64] = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
		'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
		'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/',
	};

	unsigned decode(unsigned char (&dst)[3], const unsigned char (&src)[4]){
		const void *ptr = std::memchr(BASE64_TABLE, src[0], sizeof(BASE64_TABLE));
		if(!ptr){
			DEBUG_THROW(ProtocolException, sslit("Invalid base64 character"), -1);
		}
		unsigned vals[4];
		vals[0] = (static_cast<const unsigned char *>(ptr) - BASE64_TABLE) & 0x3F;
		ptr = std::memchr(BASE64_TABLE, src[1], sizeof(BASE64_TABLE));
		if(!ptr){
			DEBUG_THROW(ProtocolException, sslit("Invalid base64 character"), -1);
		}
		vals[1] = (static_cast<const unsigned char *>(ptr) - BASE64_TABLE) & 0x3F;
		unsigned n = 1;
		if(src[2] != '='){
			ptr = std::memchr(BASE64_TABLE, src[2], sizeof(BASE64_TABLE));
			if(!ptr){
				DEBUG_THROW(ProtocolException, sslit("Invalid base64 character"), -1);
			}
			vals[2] = (static_cast<const unsigned char *>(ptr) - BASE64_TABLE) & 0x3F;
			++n;
			if(src[3] != '='){
				ptr = std::memchr(BASE64_TABLE, src[3], sizeof(BASE64_TABLE));
				if(!ptr){
					DEBUG_THROW(ProtocolException, sslit("Invalid base64 character"), -1);
				}
				vals[3] = (static_cast<const unsigned char *>(ptr) - BASE64_TABLE) & 0x3F;
				++n;
			}
		}
		if(n == 1){
			dst[0] = (vals[0] << 2) | (vals[1] >> 4);
		} else if(n == 2){
			dst[0] = (vals[0] << 2) | (vals[1] >> 4);
			dst[1] = (vals[1] << 4) | (vals[2] >> 2);
		} else {
			dst[0] = (vals[0] << 2) | (vals[1] >> 4);
			dst[1] = (vals[1] << 4) | (vals[2] >> 2);
			dst[2] = (vals[2] << 6) | (vals[3] >> 0);
		}
		return n;
	}
	void encode(unsigned char (&dst)[4], const unsigned char (&src)[3], unsigned n){
		if(n == 1){
			dst[0] = BASE64_TABLE[(                (src[0] >> 2)) & 0x3F];
			dst[1] = BASE64_TABLE[((src[0] << 4)                ) & 0x3F];
			dst[2] = '=';
			dst[3] = '=';
		} else if(n == 2){
			dst[0] = BASE64_TABLE[(                (src[0] >> 2)) & 0x3F];
			dst[1] = BASE64_TABLE[((src[0] << 4) | (src[1] >> 4)) & 0x3F];
			dst[2] = BASE64_TABLE[((src[1] << 2)                ) & 0x3F];
			dst[3] = '=';
		} else {
			dst[0] = BASE64_TABLE[(                (src[0] >> 2)) & 0x3F];
			dst[1] = BASE64_TABLE[((src[0] << 4) | (src[1] >> 4)) & 0x3F];
			dst[2] = BASE64_TABLE[((src[1] << 2) | (src[2] >> 6)) & 0x3F];
			dst[3] = BASE64_TABLE[((src[2] << 0)                ) & 0x3F];
		}
	}
}

Base64_streambuf::~Base64_streambuf(){
}

int Base64_streambuf::sync(){
	if(gptr()){
		m_buffer.discard(4);
		const unsigned n = static_cast<unsigned>(egptr() - gptr());
		if(n != 0){
			unsigned char src[3], dst[4];
			std::memcpy(src, gptr(), n);
			encode(dst, src, n);
			m_buffer.unget(dst[3]);
			m_buffer.unget(dst[2]);
			m_buffer.unget(dst[1]);
			m_buffer.unget(dst[0]);
		}
		setg(NULLPTR, NULLPTR, NULLPTR);
	}
	if(pptr()){
		const unsigned n = static_cast<unsigned>(pptr() - pbase());
		if(n != 0){
			unsigned char src[3], dst[4];
			std::memcpy(src, pbase(), n);
			encode(dst, src, n);
			m_buffer.put(dst, 4);
		}
		setp(NULLPTR, NULLPTR);
	}
	return std::streambuf::sync();
}

Base64_streambuf::int_type Base64_streambuf::underflow(){
	if(m_which & std::ios_base::in){
		sync();
		unsigned char src[4], dst[3];
		if(m_buffer.peek(src, 4) < 4){
			return traits_type::eof();
		}
		const unsigned n = decode(dst, src);
		if(n == 0){
			return traits_type::eof();
		}
		std::memcpy(m_get_area.begin(), dst, n);
		setg(m_get_area.begin(), m_get_area.begin(), m_get_area.begin() + n);
		return traits_type::to_int_type(*gptr());
	} else {
		return traits_type::eof();
	}
}

Base64_streambuf::int_type Base64_streambuf::pbackfail(Base64_streambuf::int_type c){
	if(m_which & std::ios_base::out){
		if(traits_type::eq_int_type(c, traits_type::eof())){
			return traits_type::eof();
		}
		if(gptr() && (eback() != gptr())){
			gbump(-1);
		} else if(eback() && (eback() != m_get_area.begin())){
			setg(eback() - 1, gptr() - 1, egptr());
		} else {
			sync();
			m_buffer.unget('A');
			m_buffer.unget('A');
			m_buffer.unget('A');
			m_buffer.unget('A');
			setg(m_get_area.end() - 1, m_get_area.end() - 1, m_get_area.end());
		}
		*gptr() = c;
		return c;
	} else {
		return traits_type::eof();
	}
}

Base64_streambuf::int_type Base64_streambuf::overflow(Base64_streambuf::int_type c){
	if(m_which & std::ios_base::out){
		if(traits_type::eq_int_type(c, traits_type::eof())){
			return traits_type::not_eof(c);
		}
		sync();
		setp(m_put_area.begin(), m_put_area.end());
		*pptr() = c;
		pbump(1);
		return c;
	} else {
		return traits_type::eof();
	}
}

Base64_istream::~Base64_istream(){
}

Base64_ostream::~Base64_ostream(){
}

Base64_stream::~Base64_stream(){
}

}
