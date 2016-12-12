// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "sha1.hpp"
#include "endian.hpp"

namespace Poseidon {

namespace {
	CONSTEXPR const boost::array<boost::uint32_t, 5> SHA1_REG_INIT = {{
		0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u }};

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

Sha1_streambuf::Sha1_streambuf()
	: m_reg(SHA1_REG_INIT), m_bytes(0)
{
}
Sha1_streambuf::~Sha1_streambuf(){
}

Sha1_streambuf::int_type Sha1_streambuf::overflow(Sha1_streambuf::int_type c){
	if(pptr() == m_chunk.end()){
		// https://en.wikipedia.org/wiki/SHA-1
		boost::array<boost::uint32_t, 80> ww;
		for(std::size_t i = 0; i < 16; ++i){
			ww[i] = load_be(reinterpret_cast<const boost::uint32_t *>(m_chunk.data())[i]);
		}
		for(std::size_t i = 16; i < 32; ++i){
			ww[i] = rotl<1>(ww[i - 3] ^ ww[i - 8] ^ ww[i - 14] ^ ww[i - 16]);
		}
		for(std::size_t i = 32; i < 80; ++i){
			ww[i] = rotl<2>(ww[i - 6] ^ ww[i - 16] ^ ww[i - 28] ^ ww[i - 32]);
		}

		register boost::uint32_t aa = m_reg[0];
		register boost::uint32_t bb = m_reg[1];
		register boost::uint32_t cc = m_reg[2];
		register boost::uint32_t dd = m_reg[3];
		register boost::uint32_t ee = m_reg[4];

		register boost::uint32_t ff, kk;

#define SHA1_STEP(i_, spec_, a_, b_, c_, d_, e_)	\
		spec_(a_, b_, c_, d_, e_);	\
		e_ += rotl<5>(a_) + ff + kk + ww[i_];	\
		b_ = rotl<30>(b_);

#define SHA1_SPEC_0(a_, b_, c_, d_, e_) (ff = d_ ^ (b_ & (c_ ^ d_)), kk = 0x5A827999)
#define SHA1_SPEC_1(a_, b_, c_, d_, e_) (ff = b_ ^ c_ ^ d_, kk = 0x6ED9EBA1)
#define SHA1_SPEC_2(a_, b_, c_, d_, e_) (ff = (b_ & (c_ | d_)) | (c_ & d_), kk = 0x8F1BBCDC)
#define SHA1_SPEC_3(a_, b_, c_, d_, e_) (ff = b_ ^ c_ ^ d_, kk = 0xCA62C1D6)

		SHA1_STEP( 0, SHA1_SPEC_0, aa, bb, cc, dd, ee)
		SHA1_STEP( 1, SHA1_SPEC_0, ee, aa, bb, cc, dd)
		SHA1_STEP( 2, SHA1_SPEC_0, dd, ee, aa, bb, cc)
		SHA1_STEP( 3, SHA1_SPEC_0, cc, dd, ee, aa, bb)
		SHA1_STEP( 4, SHA1_SPEC_0, bb, cc, dd, ee, aa)
		SHA1_STEP( 5, SHA1_SPEC_0, aa, bb, cc, dd, ee)
		SHA1_STEP( 6, SHA1_SPEC_0, ee, aa, bb, cc, dd)
		SHA1_STEP( 7, SHA1_SPEC_0, dd, ee, aa, bb, cc)
		SHA1_STEP( 8, SHA1_SPEC_0, cc, dd, ee, aa, bb)
		SHA1_STEP( 9, SHA1_SPEC_0, bb, cc, dd, ee, aa)
		SHA1_STEP(10, SHA1_SPEC_0, aa, bb, cc, dd, ee)
		SHA1_STEP(11, SHA1_SPEC_0, ee, aa, bb, cc, dd)
		SHA1_STEP(12, SHA1_SPEC_0, dd, ee, aa, bb, cc)
		SHA1_STEP(13, SHA1_SPEC_0, cc, dd, ee, aa, bb)
		SHA1_STEP(14, SHA1_SPEC_0, bb, cc, dd, ee, aa)
		SHA1_STEP(15, SHA1_SPEC_0, aa, bb, cc, dd, ee)
		SHA1_STEP(16, SHA1_SPEC_0, ee, aa, bb, cc, dd)
		SHA1_STEP(17, SHA1_SPEC_0, dd, ee, aa, bb, cc)
		SHA1_STEP(18, SHA1_SPEC_0, cc, dd, ee, aa, bb)
		SHA1_STEP(19, SHA1_SPEC_0, bb, cc, dd, ee, aa)

		SHA1_STEP(20, SHA1_SPEC_1, aa, bb, cc, dd, ee)
		SHA1_STEP(21, SHA1_SPEC_1, ee, aa, bb, cc, dd)
		SHA1_STEP(22, SHA1_SPEC_1, dd, ee, aa, bb, cc)
		SHA1_STEP(23, SHA1_SPEC_1, cc, dd, ee, aa, bb)
		SHA1_STEP(24, SHA1_SPEC_1, bb, cc, dd, ee, aa)
		SHA1_STEP(25, SHA1_SPEC_1, aa, bb, cc, dd, ee)
		SHA1_STEP(26, SHA1_SPEC_1, ee, aa, bb, cc, dd)
		SHA1_STEP(27, SHA1_SPEC_1, dd, ee, aa, bb, cc)
		SHA1_STEP(28, SHA1_SPEC_1, cc, dd, ee, aa, bb)
		SHA1_STEP(29, SHA1_SPEC_1, bb, cc, dd, ee, aa)
		SHA1_STEP(30, SHA1_SPEC_1, aa, bb, cc, dd, ee)
		SHA1_STEP(31, SHA1_SPEC_1, ee, aa, bb, cc, dd)
		SHA1_STEP(32, SHA1_SPEC_1, dd, ee, aa, bb, cc)
		SHA1_STEP(33, SHA1_SPEC_1, cc, dd, ee, aa, bb)
		SHA1_STEP(34, SHA1_SPEC_1, bb, cc, dd, ee, aa)
		SHA1_STEP(35, SHA1_SPEC_1, aa, bb, cc, dd, ee)
		SHA1_STEP(36, SHA1_SPEC_1, ee, aa, bb, cc, dd)
		SHA1_STEP(37, SHA1_SPEC_1, dd, ee, aa, bb, cc)
		SHA1_STEP(38, SHA1_SPEC_1, cc, dd, ee, aa, bb)
		SHA1_STEP(39, SHA1_SPEC_1, bb, cc, dd, ee, aa)

		SHA1_STEP(40, SHA1_SPEC_2, aa, bb, cc, dd, ee)
		SHA1_STEP(41, SHA1_SPEC_2, ee, aa, bb, cc, dd)
		SHA1_STEP(42, SHA1_SPEC_2, dd, ee, aa, bb, cc)
		SHA1_STEP(43, SHA1_SPEC_2, cc, dd, ee, aa, bb)
		SHA1_STEP(44, SHA1_SPEC_2, bb, cc, dd, ee, aa)
		SHA1_STEP(45, SHA1_SPEC_2, aa, bb, cc, dd, ee)
		SHA1_STEP(46, SHA1_SPEC_2, ee, aa, bb, cc, dd)
		SHA1_STEP(47, SHA1_SPEC_2, dd, ee, aa, bb, cc)
		SHA1_STEP(48, SHA1_SPEC_2, cc, dd, ee, aa, bb)
		SHA1_STEP(49, SHA1_SPEC_2, bb, cc, dd, ee, aa)
		SHA1_STEP(50, SHA1_SPEC_2, aa, bb, cc, dd, ee)
		SHA1_STEP(51, SHA1_SPEC_2, ee, aa, bb, cc, dd)
		SHA1_STEP(52, SHA1_SPEC_2, dd, ee, aa, bb, cc)
		SHA1_STEP(53, SHA1_SPEC_2, cc, dd, ee, aa, bb)
		SHA1_STEP(54, SHA1_SPEC_2, bb, cc, dd, ee, aa)
		SHA1_STEP(55, SHA1_SPEC_2, aa, bb, cc, dd, ee)
		SHA1_STEP(56, SHA1_SPEC_2, ee, aa, bb, cc, dd)
		SHA1_STEP(57, SHA1_SPEC_2, dd, ee, aa, bb, cc)
		SHA1_STEP(58, SHA1_SPEC_2, cc, dd, ee, aa, bb)
		SHA1_STEP(59, SHA1_SPEC_2, bb, cc, dd, ee, aa)

		SHA1_STEP(60, SHA1_SPEC_3, aa, bb, cc, dd, ee)
		SHA1_STEP(61, SHA1_SPEC_3, ee, aa, bb, cc, dd)
		SHA1_STEP(62, SHA1_SPEC_3, dd, ee, aa, bb, cc)
		SHA1_STEP(63, SHA1_SPEC_3, cc, dd, ee, aa, bb)
		SHA1_STEP(64, SHA1_SPEC_3, bb, cc, dd, ee, aa)
		SHA1_STEP(65, SHA1_SPEC_3, aa, bb, cc, dd, ee)
		SHA1_STEP(66, SHA1_SPEC_3, ee, aa, bb, cc, dd)
		SHA1_STEP(67, SHA1_SPEC_3, dd, ee, aa, bb, cc)
		SHA1_STEP(68, SHA1_SPEC_3, cc, dd, ee, aa, bb)
		SHA1_STEP(69, SHA1_SPEC_3, bb, cc, dd, ee, aa)
		SHA1_STEP(70, SHA1_SPEC_3, aa, bb, cc, dd, ee)
		SHA1_STEP(71, SHA1_SPEC_3, ee, aa, bb, cc, dd)
		SHA1_STEP(72, SHA1_SPEC_3, dd, ee, aa, bb, cc)
		SHA1_STEP(73, SHA1_SPEC_3, cc, dd, ee, aa, bb)
		SHA1_STEP(74, SHA1_SPEC_3, bb, cc, dd, ee, aa)
		SHA1_STEP(75, SHA1_SPEC_3, aa, bb, cc, dd, ee)
		SHA1_STEP(76, SHA1_SPEC_3, ee, aa, bb, cc, dd)
		SHA1_STEP(77, SHA1_SPEC_3, dd, ee, aa, bb, cc)
		SHA1_STEP(78, SHA1_SPEC_3, cc, dd, ee, aa, bb)
		SHA1_STEP(79, SHA1_SPEC_3, bb, cc, dd, ee, aa)

		m_reg[0] += aa;
		m_reg[1] += bb;
		m_reg[2] += cc;
		m_reg[3] += dd;
		m_reg[4] += ee;
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

Sha1 Sha1_streambuf::finalize(){
	boost::uint64_t bytes = m_bytes;
	if(pptr()){
		bytes += static_cast<unsigned>(pptr() - m_chunk.begin());
	}
	sputc(0x80);
	while(pptr() != m_chunk.begin() + 56){
		sputc(0);
	}
	boost::uint64_t bits;
	store_be(bits, bytes * 8);
	xsputn(reinterpret_cast<const char *>(&bits), sizeof(bits));
	overflow(traits_type::eof());

	Sha1 sha1;
	for(unsigned i = 0; i < m_reg.size(); ++i){
		store_be(reinterpret_cast<boost::uint32_t *>(sha1.data())[i], m_reg[i]);
	}
	m_reg = SHA1_REG_INIT;
	m_bytes = 0;
	return sha1;
}

Sha1_ostream::~Sha1_ostream(){
}

}
