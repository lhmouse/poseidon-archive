// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "md5.hpp"
#include "endian.hpp"

namespace Poseidon {

namespace {
	CONSTEXPR const boost::array<boost::uint32_t, 4> MD5_REG_INIT = {{
		0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u }};

	template<unsigned N>
	inline boost::uint32_t rotl(boost::uint32_t u){
		boost::uint32_t r = u;
		__asm__("roll %1, %0 \n" : "+r"(r) : "I"(N));
		return r;
	}
	template<unsigned N>
	inline boost::uint32_t rotr(boost::uint32_t u){
		boost::uint32_t r = u;
		__asm__("rorl %1, %0 \n" : "+r"(r) : "I"(N));
		return r;
	}
}

Md5_streambuf::Md5_streambuf()
	: m_reg(MD5_REG_INIT), m_bytes(0)
{
}
Md5_streambuf::~Md5_streambuf(){
}

Md5_streambuf::int_type Md5_streambuf::overflow(Md5_streambuf::int_type c){
	if(pptr() == m_chunk.end()){
		// https://en.wikipedia.org/wiki/MD5
		AUTO_REF(ww, *reinterpret_cast<boost::uint32_t (*)[4]>(m_chunk.data()));

		register boost::uint32_t aa = m_reg[0];
		register boost::uint32_t bb = m_reg[1];
		register boost::uint32_t cc = m_reg[2];
		register boost::uint32_t dd = m_reg[3];

		register boost::uint32_t ff, gg;

#define MD5_STEP(i_, spec_, a_, b_, c_, d_, k_, r_)	\
		spec_(i_, a_, b_, c_, d_);	\
		a_ = b_ + rotl<r_>(a_ + ff + k_ + load_le(ww[gg]));

#define MD5_SPEC_0(i_, a_, b_, c_, d_)  (ff = d_ ^ (b_ & (c_ ^ d_)), gg = i_)
#define MD5_SPEC_1(i_, a_, b_, c_, d_)  (ff = c_ ^ (d_ & (b_ ^ c_)), gg = (5 * i_ + 1) % 16)
#define MD5_SPEC_2(i_, a_, b_, c_, d_)  (ff = b_ ^ c_ ^ d_, gg = (3 * i_ + 5) % 16)
#define MD5_SPEC_3(i_, a_, b_, c_, d_)  (ff = c_ ^ (b_ | ~d_), gg = (7 * i_) % 16)

		MD5_STEP( 0, MD5_SPEC_0, aa, bb, cc, dd, 0xD76AA478,  7)
		MD5_STEP( 1, MD5_SPEC_0, dd, aa, bb, cc, 0xE8C7B756, 12)
		MD5_STEP( 2, MD5_SPEC_0, cc, dd, aa, bb, 0x242070DB, 17)
		MD5_STEP( 3, MD5_SPEC_0, bb, cc, dd, aa, 0xC1BDCEEE, 22)
		MD5_STEP( 4, MD5_SPEC_0, aa, bb, cc, dd, 0xF57C0FAF,  7)
		MD5_STEP( 5, MD5_SPEC_0, dd, aa, bb, cc, 0x4787C62A, 12)
		MD5_STEP( 6, MD5_SPEC_0, cc, dd, aa, bb, 0xA8304613, 17)
		MD5_STEP( 7, MD5_SPEC_0, bb, cc, dd, aa, 0xFD469501, 22)
		MD5_STEP( 8, MD5_SPEC_0, aa, bb, cc, dd, 0x698098D8,  7)
		MD5_STEP( 9, MD5_SPEC_0, dd, aa, bb, cc, 0x8B44F7AF, 12)
		MD5_STEP(10, MD5_SPEC_0, cc, dd, aa, bb, 0xFFFF5BB1, 17)
		MD5_STEP(11, MD5_SPEC_0, bb, cc, dd, aa, 0x895CD7BE, 22)
		MD5_STEP(12, MD5_SPEC_0, aa, bb, cc, dd, 0x6B901122,  7)
		MD5_STEP(13, MD5_SPEC_0, dd, aa, bb, cc, 0xFD987193, 12)
		MD5_STEP(14, MD5_SPEC_0, cc, dd, aa, bb, 0xA679438E, 17)
		MD5_STEP(15, MD5_SPEC_0, bb, cc, dd, aa, 0x49B40821, 22)

		MD5_STEP(16, MD5_SPEC_1, aa, bb, cc, dd, 0xF61E2562,  5)
		MD5_STEP(17, MD5_SPEC_1, dd, aa, bb, cc, 0xC040B340,  9)
		MD5_STEP(18, MD5_SPEC_1, cc, dd, aa, bb, 0x265E5A51, 14)
		MD5_STEP(19, MD5_SPEC_1, bb, cc, dd, aa, 0xE9B6C7AA, 20)
		MD5_STEP(20, MD5_SPEC_1, aa, bb, cc, dd, 0xD62F105D,  5)
		MD5_STEP(21, MD5_SPEC_1, dd, aa, bb, cc, 0x02441453,  9)
		MD5_STEP(22, MD5_SPEC_1, cc, dd, aa, bb, 0xD8A1E681, 14)
		MD5_STEP(23, MD5_SPEC_1, bb, cc, dd, aa, 0xE7D3FBC8, 20)
		MD5_STEP(24, MD5_SPEC_1, aa, bb, cc, dd, 0x21E1CDE6,  5)
		MD5_STEP(25, MD5_SPEC_1, dd, aa, bb, cc, 0xC33707D6,  9)
		MD5_STEP(26, MD5_SPEC_1, cc, dd, aa, bb, 0xF4D50D87, 14)
		MD5_STEP(27, MD5_SPEC_1, bb, cc, dd, aa, 0x455A14ED, 20)
		MD5_STEP(28, MD5_SPEC_1, aa, bb, cc, dd, 0xA9E3E905,  5)
		MD5_STEP(29, MD5_SPEC_1, dd, aa, bb, cc, 0xFCEFA3F8,  9)
		MD5_STEP(30, MD5_SPEC_1, cc, dd, aa, bb, 0x676F02D9, 14)
		MD5_STEP(31, MD5_SPEC_1, bb, cc, dd, aa, 0x8D2A4C8A, 20)

		MD5_STEP(32, MD5_SPEC_2, aa, bb, cc, dd, 0xFFFA3942,  4)
		MD5_STEP(33, MD5_SPEC_2, dd, aa, bb, cc, 0x8771F681, 11)
		MD5_STEP(34, MD5_SPEC_2, cc, dd, aa, bb, 0x6D9D6122, 16)
		MD5_STEP(35, MD5_SPEC_2, bb, cc, dd, aa, 0xFDE5380C, 23)
		MD5_STEP(36, MD5_SPEC_2, aa, bb, cc, dd, 0xA4BEEA44,  4)
		MD5_STEP(37, MD5_SPEC_2, dd, aa, bb, cc, 0x4BDECFA9, 11)
		MD5_STEP(38, MD5_SPEC_2, cc, dd, aa, bb, 0xF6BB4B60, 16)
		MD5_STEP(39, MD5_SPEC_2, bb, cc, dd, aa, 0xBEBFBC70, 23)
		MD5_STEP(40, MD5_SPEC_2, aa, bb, cc, dd, 0x289B7EC6,  4)
		MD5_STEP(41, MD5_SPEC_2, dd, aa, bb, cc, 0xEAA127FA, 11)
		MD5_STEP(42, MD5_SPEC_2, cc, dd, aa, bb, 0xD4EF3085, 16)
		MD5_STEP(43, MD5_SPEC_2, bb, cc, dd, aa, 0x04881D05, 23)
		MD5_STEP(44, MD5_SPEC_2, aa, bb, cc, dd, 0xD9D4D039,  4)
		MD5_STEP(45, MD5_SPEC_2, dd, aa, bb, cc, 0xE6DB99E5, 11)
		MD5_STEP(46, MD5_SPEC_2, cc, dd, aa, bb, 0x1FA27CF8, 16)
		MD5_STEP(47, MD5_SPEC_2, bb, cc, dd, aa, 0xC4AC5665, 23)

		MD5_STEP(48, MD5_SPEC_3, aa, bb, cc, dd, 0xF4292244,  6)
		MD5_STEP(49, MD5_SPEC_3, dd, aa, bb, cc, 0x432AFF97, 10)
		MD5_STEP(50, MD5_SPEC_3, cc, dd, aa, bb, 0xAB9423A7, 15)
		MD5_STEP(51, MD5_SPEC_3, bb, cc, dd, aa, 0xFC93A039, 21)
		MD5_STEP(52, MD5_SPEC_3, aa, bb, cc, dd, 0x655B59C3,  6)
		MD5_STEP(53, MD5_SPEC_3, dd, aa, bb, cc, 0x8F0CCC92, 10)
		MD5_STEP(54, MD5_SPEC_3, cc, dd, aa, bb, 0xFFEFF47D, 15)
		MD5_STEP(55, MD5_SPEC_3, bb, cc, dd, aa, 0x85845DD1, 21)
		MD5_STEP(56, MD5_SPEC_3, aa, bb, cc, dd, 0x6FA87E4F,  6)
		MD5_STEP(57, MD5_SPEC_3, dd, aa, bb, cc, 0xFE2CE6E0, 10)
		MD5_STEP(58, MD5_SPEC_3, cc, dd, aa, bb, 0xA3014314, 15)
		MD5_STEP(59, MD5_SPEC_3, bb, cc, dd, aa, 0x4E0811A1, 21)
		MD5_STEP(60, MD5_SPEC_3, aa, bb, cc, dd, 0xF7537E82,  6)
		MD5_STEP(61, MD5_SPEC_3, dd, aa, bb, cc, 0xBD3AF235, 10)
		MD5_STEP(62, MD5_SPEC_3, cc, dd, aa, bb, 0x2AD7D2BB, 15)
		MD5_STEP(63, MD5_SPEC_3, bb, cc, dd, aa, 0xEB86D391, 21)

		m_reg[0] += aa;
		m_reg[1] += bb;
		m_reg[2] += cc;
		m_reg[3] += dd;
		m_bytes += 64;

		setp(m_chunk.begin(), m_chunk.end());
	}
	if(traits_type::eq_int_type(c, traits_type::eof())){
		return traits_type::not_eof(c);
	}
	setp(m_chunk.begin() + 1, m_chunk.end());
	pptr()[-1] = c;
	return c;
}

Md5 Md5_streambuf::finalize(){
	boost::uint64_t bytes = m_bytes;
	if(pptr()){
		bytes += static_cast<unsigned>(pptr() - m_chunk.begin());
	}
	sputc(0x80);
	while(pptr() != m_chunk.begin() + 56){
		sputc(0);
	}
	boost::uint64_t bits;
	store_le(bits, bytes * 8);
	xsputn(reinterpret_cast<const char *>(&bits), sizeof(bits));
	overflow(traits_type::eof());

	Md5 md5;
	for(unsigned i = 0; i < m_reg.size(); ++i){
		store_le(reinterpret_cast<boost::uint32_t *>(md5.data())[i], m_reg[i]);
	}
	m_reg = MD5_REG_INIT;
	m_bytes = 0;
	return md5;
}

Md5_ostream::~Md5_ostream(){
}

}
