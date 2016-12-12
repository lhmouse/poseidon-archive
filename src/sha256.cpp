// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "sha256.hpp"
#include "endian.hpp"

namespace Poseidon {

namespace {
	CONSTEXPR const boost::array<boost::uint32_t, 8> SHA256_REG_INIT = {{
		0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u }};

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

Sha256_streambuf::Sha256_streambuf()
	: m_reg(SHA256_REG_INIT), m_bytes(0)
{
}
Sha256_streambuf::~Sha256_streambuf(){
}

Sha256_streambuf::int_type Sha256_streambuf::overflow(Sha256_streambuf::int_type c){
	if(pptr() == m_chunk.end()){
		// https://en.wikipedia.org/wiki/SHA-2
		boost::array<boost::uint32_t, 64> ww;
		for(std::size_t i = 0; i < 16; ++i){
			ww[i] = load_be(reinterpret_cast<const boost::uint32_t *>(m_chunk.data())[i]);
		}
		for(std::size_t i = 16; i < 64; ++i){
			const boost::uint32_t s0 = rotr<7>((rotr<11>(ww[i - 15]) ^ ww[i - 15])) ^ (ww[i - 15] >> 3);
			const boost::uint32_t s1 = rotr<17>((rotr<2>(ww[i - 2]) ^ ww[i - 2])) ^ (ww[i - 2] >> 10);
			ww[i] = ww[i - 16] + ww[i - 7] + s0 + s1;
		}

		register boost::uint32_t aa = m_reg[0];
		register boost::uint32_t bb = m_reg[1];
		register boost::uint32_t cc = m_reg[2];
		register boost::uint32_t dd = m_reg[3];
		register boost::uint32_t ee = m_reg[4];
		register boost::uint32_t ff = m_reg[5];
		register boost::uint32_t gg = m_reg[6];
		register boost::uint32_t hh = m_reg[7];

		register boost::uint32_t S0, maj, t2, S1, ch, t1;

#define SHA256_STEP(i_, a_, b_, c_, d_, e_, f_, g_, h_, k_)	\
		S0 = rotr<2>(rotr<11>(rotr<9>(a_) ^ a_) ^ a_);	\
		maj = (a_ & b_) | (c_ & (a_ ^ b_));	\
		t2 = S0 + maj;	\
		S1 = rotr<6>(rotr<5>(rotr<14>(e_) ^ e_) ^ e_);	\
		ch = g_ ^ (e_ & (f_ ^ g_));	\
		t1 = h_ + S1 + ch + k_ + ww[i_];	\
		d_ += t1;	\
		h_ = t1 + t2;

		SHA256_STEP( 0, aa, bb, cc, dd, ee, ff, gg, hh, 0x428A2F98)
		SHA256_STEP( 1, hh, aa, bb, cc, dd, ee, ff, gg, 0x71374491)
		SHA256_STEP( 2, gg, hh, aa, bb, cc, dd, ee, ff, 0xB5C0FBCF)
		SHA256_STEP( 3, ff, gg, hh, aa, bb, cc, dd, ee, 0xE9B5DBA5)
		SHA256_STEP( 4, ee, ff, gg, hh, aa, bb, cc, dd, 0x3956C25B)
		SHA256_STEP( 5, dd, ee, ff, gg, hh, aa, bb, cc, 0x59F111F1)
		SHA256_STEP( 6, cc, dd, ee, ff, gg, hh, aa, bb, 0x923F82A4)
		SHA256_STEP( 7, bb, cc, dd, ee, ff, gg, hh, aa, 0xAB1C5ED5)

		SHA256_STEP( 8, aa, bb, cc, dd, ee, ff, gg, hh, 0xD807AA98)
		SHA256_STEP( 9, hh, aa, bb, cc, dd, ee, ff, gg, 0x12835B01)
		SHA256_STEP(10, gg, hh, aa, bb, cc, dd, ee, ff, 0x243185BE)
		SHA256_STEP(11, ff, gg, hh, aa, bb, cc, dd, ee, 0x550C7DC3)
		SHA256_STEP(12, ee, ff, gg, hh, aa, bb, cc, dd, 0x72BE5D74)
		SHA256_STEP(13, dd, ee, ff, gg, hh, aa, bb, cc, 0x80DEB1FE)
		SHA256_STEP(14, cc, dd, ee, ff, gg, hh, aa, bb, 0x9BDC06A7)
		SHA256_STEP(15, bb, cc, dd, ee, ff, gg, hh, aa, 0xC19BF174)

		SHA256_STEP(16, aa, bb, cc, dd, ee, ff, gg, hh, 0xE49B69C1)
		SHA256_STEP(17, hh, aa, bb, cc, dd, ee, ff, gg, 0xEFBE4786)
		SHA256_STEP(18, gg, hh, aa, bb, cc, dd, ee, ff, 0x0FC19DC6)
		SHA256_STEP(19, ff, gg, hh, aa, bb, cc, dd, ee, 0x240CA1CC)
		SHA256_STEP(20, ee, ff, gg, hh, aa, bb, cc, dd, 0x2DE92C6F)
		SHA256_STEP(21, dd, ee, ff, gg, hh, aa, bb, cc, 0x4A7484AA)
		SHA256_STEP(22, cc, dd, ee, ff, gg, hh, aa, bb, 0x5CB0A9DC)
		SHA256_STEP(23, bb, cc, dd, ee, ff, gg, hh, aa, 0x76F988DA)

		SHA256_STEP(24, aa, bb, cc, dd, ee, ff, gg, hh, 0x983E5152)
		SHA256_STEP(25, hh, aa, bb, cc, dd, ee, ff, gg, 0xA831C66D)
		SHA256_STEP(26, gg, hh, aa, bb, cc, dd, ee, ff, 0xB00327C8)
		SHA256_STEP(27, ff, gg, hh, aa, bb, cc, dd, ee, 0xBF597FC7)
		SHA256_STEP(28, ee, ff, gg, hh, aa, bb, cc, dd, 0xC6E00BF3)
		SHA256_STEP(29, dd, ee, ff, gg, hh, aa, bb, cc, 0xD5A79147)
		SHA256_STEP(30, cc, dd, ee, ff, gg, hh, aa, bb, 0x06CA6351)
		SHA256_STEP(31, bb, cc, dd, ee, ff, gg, hh, aa, 0x14292967)

		SHA256_STEP(32, aa, bb, cc, dd, ee, ff, gg, hh, 0x27B70A85)
		SHA256_STEP(33, hh, aa, bb, cc, dd, ee, ff, gg, 0x2E1B2138)
		SHA256_STEP(34, gg, hh, aa, bb, cc, dd, ee, ff, 0x4D2C6DFC)
		SHA256_STEP(35, ff, gg, hh, aa, bb, cc, dd, ee, 0x53380D13)
		SHA256_STEP(36, ee, ff, gg, hh, aa, bb, cc, dd, 0x650A7354)
		SHA256_STEP(37, dd, ee, ff, gg, hh, aa, bb, cc, 0x766A0ABB)
		SHA256_STEP(38, cc, dd, ee, ff, gg, hh, aa, bb, 0x81C2C92E)
		SHA256_STEP(39, bb, cc, dd, ee, ff, gg, hh, aa, 0x92722C85)

		SHA256_STEP(40, aa, bb, cc, dd, ee, ff, gg, hh, 0xA2BFE8A1)
		SHA256_STEP(41, hh, aa, bb, cc, dd, ee, ff, gg, 0xA81A664B)
		SHA256_STEP(42, gg, hh, aa, bb, cc, dd, ee, ff, 0xC24B8B70)
		SHA256_STEP(43, ff, gg, hh, aa, bb, cc, dd, ee, 0xC76C51A3)
		SHA256_STEP(44, ee, ff, gg, hh, aa, bb, cc, dd, 0xD192E819)
		SHA256_STEP(45, dd, ee, ff, gg, hh, aa, bb, cc, 0xD6990624)
		SHA256_STEP(46, cc, dd, ee, ff, gg, hh, aa, bb, 0xF40E3585)
		SHA256_STEP(47, bb, cc, dd, ee, ff, gg, hh, aa, 0x106AA070)

		SHA256_STEP(48, aa, bb, cc, dd, ee, ff, gg, hh, 0x19A4C116)
		SHA256_STEP(49, hh, aa, bb, cc, dd, ee, ff, gg, 0x1E376C08)
		SHA256_STEP(50, gg, hh, aa, bb, cc, dd, ee, ff, 0x2748774C)
		SHA256_STEP(51, ff, gg, hh, aa, bb, cc, dd, ee, 0x34B0BCB5)
		SHA256_STEP(52, ee, ff, gg, hh, aa, bb, cc, dd, 0x391C0CB3)
		SHA256_STEP(53, dd, ee, ff, gg, hh, aa, bb, cc, 0x4ED8AA4A)
		SHA256_STEP(54, cc, dd, ee, ff, gg, hh, aa, bb, 0x5B9CCA4F)
		SHA256_STEP(55, bb, cc, dd, ee, ff, gg, hh, aa, 0x682E6FF3)

		SHA256_STEP(56, aa, bb, cc, dd, ee, ff, gg, hh, 0x748F82EE)
		SHA256_STEP(57, hh, aa, bb, cc, dd, ee, ff, gg, 0x78A5636F)
		SHA256_STEP(58, gg, hh, aa, bb, cc, dd, ee, ff, 0x84C87814)
		SHA256_STEP(59, ff, gg, hh, aa, bb, cc, dd, ee, 0x8CC70208)
		SHA256_STEP(60, ee, ff, gg, hh, aa, bb, cc, dd, 0x90BEFFFA)
		SHA256_STEP(61, dd, ee, ff, gg, hh, aa, bb, cc, 0xA4506CEB)
		SHA256_STEP(62, cc, dd, ee, ff, gg, hh, aa, bb, 0xBEF9A3F7)
		SHA256_STEP(63, bb, cc, dd, ee, ff, gg, hh, aa, 0xC67178F2)

		m_reg[0] += aa;
		m_reg[1] += bb;
		m_reg[2] += cc;
		m_reg[3] += dd;
		m_reg[4] += ee;
		m_reg[5] += ff;
		m_reg[6] += gg;
		m_reg[7] += hh;
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

Sha256 Sha256_streambuf::finalize(){
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

	Sha256 sha256;
	for(unsigned i = 0; i < m_reg.size(); ++i){
		store_be(reinterpret_cast<boost::uint32_t *>(sha256.data())[i], m_reg[i]);
	}
	m_reg = SHA256_REG_INIT;
	m_bytes = 0;
	return sha256;
}

Sha256_ostream::~Sha256_ostream(){
}

}
