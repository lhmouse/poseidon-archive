// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "md5.hpp"
#include "endian.hpp"
#include <x86intrin.h>

namespace Poseidon {

namespace {
	CONSTEXPR const boost::array<boost::uint32_t, 4> g_md5_reg_init = {{ 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u }};
}

Md5_streambuf::Md5_streambuf()
	: m_reg(g_md5_reg_init), m_bytes(0)
{
	//
}
Md5_streambuf::~Md5_streambuf(){
	//
}

void Md5_streambuf::eat_chunk(){
	// https://en.wikipedia.org/wiki/MD5
	AUTO_REF(w, *reinterpret_cast<boost::uint32_t (*)[4]>(m_chunk.data()));

	register boost::uint32_t a = m_reg[0];
	register boost::uint32_t b = m_reg[1];
	register boost::uint32_t c = m_reg[2];
	register boost::uint32_t d = m_reg[3];

	register boost::uint32_t f, g;

#define MD5_STEP(i_, spec_, a_, b_, c_, d_, k_, r_)	\
	spec_(i_, a_, b_, c_, d_);	\
	a_ = b_ + __rold(a_ + f + k_ + load_le(w[g]), r_);

#define MD5_SPEC_0(i_, a_, b_, c_, d_)  (f = d_ ^ (b_ & (c_ ^ d_)), g = i_)
#define MD5_SPEC_1(i_, a_, b_, c_, d_)  (f = c_ ^ (d_ & (b_ ^ c_)), g = (5 * i_ + 1) % 16)
#define MD5_SPEC_2(i_, a_, b_, c_, d_)  (f = b_ ^ c_ ^ d_, g = (3 * i_ + 5) % 16)
#define MD5_SPEC_3(i_, a_, b_, c_, d_)  (f = c_ ^ (b_ | ~d_), g = (7 * i_) % 16)

	MD5_STEP( 0, MD5_SPEC_0, a, b, c, d, 0xD76AA478,  7)
	MD5_STEP( 1, MD5_SPEC_0, d, a, b, c, 0xE8C7B756, 12)
	MD5_STEP( 2, MD5_SPEC_0, c, d, a, b, 0x242070DB, 17)
	MD5_STEP( 3, MD5_SPEC_0, b, c, d, a, 0xC1BDCEEE, 22)
	MD5_STEP( 4, MD5_SPEC_0, a, b, c, d, 0xF57C0FAF,  7)
	MD5_STEP( 5, MD5_SPEC_0, d, a, b, c, 0x4787C62A, 12)
	MD5_STEP( 6, MD5_SPEC_0, c, d, a, b, 0xA8304613, 17)
	MD5_STEP( 7, MD5_SPEC_0, b, c, d, a, 0xFD469501, 22)
	MD5_STEP( 8, MD5_SPEC_0, a, b, c, d, 0x698098D8,  7)
	MD5_STEP( 9, MD5_SPEC_0, d, a, b, c, 0x8B44F7AF, 12)
	MD5_STEP(10, MD5_SPEC_0, c, d, a, b, 0xFFFF5BB1, 17)
	MD5_STEP(11, MD5_SPEC_0, b, c, d, a, 0x895CD7BE, 22)
	MD5_STEP(12, MD5_SPEC_0, a, b, c, d, 0x6B901122,  7)
	MD5_STEP(13, MD5_SPEC_0, d, a, b, c, 0xFD987193, 12)
	MD5_STEP(14, MD5_SPEC_0, c, d, a, b, 0xA679438E, 17)
	MD5_STEP(15, MD5_SPEC_0, b, c, d, a, 0x49B40821, 22)

	MD5_STEP(16, MD5_SPEC_1, a, b, c, d, 0xF61E2562,  5)
	MD5_STEP(17, MD5_SPEC_1, d, a, b, c, 0xC040B340,  9)
	MD5_STEP(18, MD5_SPEC_1, c, d, a, b, 0x265E5A51, 14)
	MD5_STEP(19, MD5_SPEC_1, b, c, d, a, 0xE9B6C7AA, 20)
	MD5_STEP(20, MD5_SPEC_1, a, b, c, d, 0xD62F105D,  5)
	MD5_STEP(21, MD5_SPEC_1, d, a, b, c, 0x02441453,  9)
	MD5_STEP(22, MD5_SPEC_1, c, d, a, b, 0xD8A1E681, 14)
	MD5_STEP(23, MD5_SPEC_1, b, c, d, a, 0xE7D3FBC8, 20)
	MD5_STEP(24, MD5_SPEC_1, a, b, c, d, 0x21E1CDE6,  5)
	MD5_STEP(25, MD5_SPEC_1, d, a, b, c, 0xC33707D6,  9)
	MD5_STEP(26, MD5_SPEC_1, c, d, a, b, 0xF4D50D87, 14)
	MD5_STEP(27, MD5_SPEC_1, b, c, d, a, 0x455A14ED, 20)
	MD5_STEP(28, MD5_SPEC_1, a, b, c, d, 0xA9E3E905,  5)
	MD5_STEP(29, MD5_SPEC_1, d, a, b, c, 0xFCEFA3F8,  9)
	MD5_STEP(30, MD5_SPEC_1, c, d, a, b, 0x676F02D9, 14)
	MD5_STEP(31, MD5_SPEC_1, b, c, d, a, 0x8D2A4C8A, 20)

	MD5_STEP(32, MD5_SPEC_2, a, b, c, d, 0xFFFA3942,  4)
	MD5_STEP(33, MD5_SPEC_2, d, a, b, c, 0x8771F681, 11)
	MD5_STEP(34, MD5_SPEC_2, c, d, a, b, 0x6D9D6122, 16)
	MD5_STEP(35, MD5_SPEC_2, b, c, d, a, 0xFDE5380C, 23)
	MD5_STEP(36, MD5_SPEC_2, a, b, c, d, 0xA4BEEA44,  4)
	MD5_STEP(37, MD5_SPEC_2, d, a, b, c, 0x4BDECFA9, 11)
	MD5_STEP(38, MD5_SPEC_2, c, d, a, b, 0xF6BB4B60, 16)
	MD5_STEP(39, MD5_SPEC_2, b, c, d, a, 0xBEBFBC70, 23)
	MD5_STEP(40, MD5_SPEC_2, a, b, c, d, 0x289B7EC6,  4)
	MD5_STEP(41, MD5_SPEC_2, d, a, b, c, 0xEAA127FA, 11)
	MD5_STEP(42, MD5_SPEC_2, c, d, a, b, 0xD4EF3085, 16)
	MD5_STEP(43, MD5_SPEC_2, b, c, d, a, 0x04881D05, 23)
	MD5_STEP(44, MD5_SPEC_2, a, b, c, d, 0xD9D4D039,  4)
	MD5_STEP(45, MD5_SPEC_2, d, a, b, c, 0xE6DB99E5, 11)
	MD5_STEP(46, MD5_SPEC_2, c, d, a, b, 0x1FA27CF8, 16)
	MD5_STEP(47, MD5_SPEC_2, b, c, d, a, 0xC4AC5665, 23)

	MD5_STEP(48, MD5_SPEC_3, a, b, c, d, 0xF4292244,  6)
	MD5_STEP(49, MD5_SPEC_3, d, a, b, c, 0x432AFF97, 10)
	MD5_STEP(50, MD5_SPEC_3, c, d, a, b, 0xAB9423A7, 15)
	MD5_STEP(51, MD5_SPEC_3, b, c, d, a, 0xFC93A039, 21)
	MD5_STEP(52, MD5_SPEC_3, a, b, c, d, 0x655B59C3,  6)
	MD5_STEP(53, MD5_SPEC_3, d, a, b, c, 0x8F0CCC92, 10)
	MD5_STEP(54, MD5_SPEC_3, c, d, a, b, 0xFFEFF47D, 15)
	MD5_STEP(55, MD5_SPEC_3, b, c, d, a, 0x85845DD1, 21)
	MD5_STEP(56, MD5_SPEC_3, a, b, c, d, 0x6FA87E4F,  6)
	MD5_STEP(57, MD5_SPEC_3, d, a, b, c, 0xFE2CE6E0, 10)
	MD5_STEP(58, MD5_SPEC_3, c, d, a, b, 0xA3014314, 15)
	MD5_STEP(59, MD5_SPEC_3, b, c, d, a, 0x4E0811A1, 21)
	MD5_STEP(60, MD5_SPEC_3, a, b, c, d, 0xF7537E82,  6)
	MD5_STEP(61, MD5_SPEC_3, d, a, b, c, 0xBD3AF235, 10)
	MD5_STEP(62, MD5_SPEC_3, c, d, a, b, 0x2AD7D2BB, 15)
	MD5_STEP(63, MD5_SPEC_3, b, c, d, a, 0xEB86D391, 21)

	m_reg[0] += a;
	m_reg[1] += b;
	m_reg[2] += c;
	m_reg[3] += d;
	m_bytes += 64;
}

void Md5_streambuf::reset() NOEXCEPT {
	setp(NULLPTR, NULLPTR);
	m_reg = g_md5_reg_init;
	m_bytes = 0;
}
Md5_streambuf::int_type Md5_streambuf::overflow(Md5_streambuf::int_type c){
	if(pptr() == m_chunk.end()){
		eat_chunk();
		setp(NULLPTR, NULLPTR);
	}
	if(traits_type::eq_int_type(c, traits_type::eof())){
		return traits_type::not_eof(c);
	}
	setp(m_chunk.begin(), m_chunk.end());
	*pptr() = traits_type::to_char_type(c);
	pbump(1);
	return c;
}

Md5 Md5_streambuf::finalize(){
	boost::uint64_t bytes = m_bytes;
	if(pptr()){
		bytes += static_cast<unsigned>(pptr() - m_chunk.begin());
	}
	sputc(traits_type::to_char_type(0x80));
	while(pptr() != m_chunk.begin() + 56){
		sputc(traits_type::to_char_type(0));
	}
	boost::uint64_t bits;
	store_le(bits, bytes * 8);
	xsputn(reinterpret_cast<const char *>(&bits), sizeof(bits));
	if(pptr() == m_chunk.end()){
		eat_chunk();
	}

	Md5 md5;
	for(unsigned i = 0; i < m_reg.size(); ++i){
		store_le(reinterpret_cast<boost::uint32_t *>(md5.data())[i], m_reg[i]);
	}
	reset();
	return md5;
}

Md5_ostream::~Md5_ostream(){
	//
}

}
