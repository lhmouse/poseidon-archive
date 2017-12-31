// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "sha256.hpp"
#include "endian.hpp"
#include <x86intrin.h>

namespace Poseidon {

namespace {
	CONSTEXPR const boost::array<boost::uint32_t, 8> SHA256_REG_INIT = {{ 0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u }};
}

Sha256_streambuf::Sha256_streambuf()
	: m_reg(SHA256_REG_INIT), m_bytes(0)
{ }
Sha256_streambuf::~Sha256_streambuf(){ }

void Sha256_streambuf::eat_chunk(){
	// https://en.wikipedia.org/wiki/SHA-2
	boost::array<boost::uint32_t, 64> w;
	for(std::size_t i = 0; i < 16; ++i){
		w[i] = load_be(reinterpret_cast<const boost::uint32_t *>(m_chunk.data())[i]);
	}
	for(std::size_t i = 16; i < 64; ++i){
		const boost::uint32_t s0 = __rord(__rord(w[i - 15], 11) ^ w[i - 15], 7) ^ (w[i - 15] >> 3);
		const boost::uint32_t s1 = __rord(__rord(w[i - 2], 2) ^ w[i - 2], 17) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + w[i - 7] + s0 + s1;
	}

	register boost::uint32_t a = m_reg[0];
	register boost::uint32_t b = m_reg[1];
	register boost::uint32_t c = m_reg[2];
	register boost::uint32_t d = m_reg[3];
	register boost::uint32_t e = m_reg[4];
	register boost::uint32_t f = m_reg[5];
	register boost::uint32_t g = m_reg[6];
	register boost::uint32_t h = m_reg[7];

	register boost::uint32_t S0, maj, t2, S1, ch, t1;

#define SHA256_STEP(i_, a_, b_, c_, d_, e_, f_, g_, h_, k_)	\
	S0 = __rord(__rord(__rord(a_, 9) ^ a_, 11) ^ a_, 2);	\
	maj = (a_ & b_) | (c_ & (a_ ^ b_));	\
	t2 = S0 + maj;	\
	S1 = __rord(__rord(__rord(e_, 14) ^ e_, 5) ^ e_, 6);	\
	ch = g_ ^ (e_ & (f_ ^ g_));	\
	t1 = h_ + S1 + ch + k_ + w[i_];	\
	d_ += t1;	\
	h_ = t1 + t2;

	SHA256_STEP( 0, a, b, c, d, e, f, g, h, 0x428A2F98)
	SHA256_STEP( 1, h, a, b, c, d, e, f, g, 0x71374491)
	SHA256_STEP( 2, g, h, a, b, c, d, e, f, 0xB5C0FBCF)
	SHA256_STEP( 3, f, g, h, a, b, c, d, e, 0xE9B5DBA5)
	SHA256_STEP( 4, e, f, g, h, a, b, c, d, 0x3956C25B)
	SHA256_STEP( 5, d, e, f, g, h, a, b, c, 0x59F111F1)
	SHA256_STEP( 6, c, d, e, f, g, h, a, b, 0x923F82A4)
	SHA256_STEP( 7, b, c, d, e, f, g, h, a, 0xAB1C5ED5)

	SHA256_STEP( 8, a, b, c, d, e, f, g, h, 0xD807AA98)
	SHA256_STEP( 9, h, a, b, c, d, e, f, g, 0x12835B01)
	SHA256_STEP(10, g, h, a, b, c, d, e, f, 0x243185BE)
	SHA256_STEP(11, f, g, h, a, b, c, d, e, 0x550C7DC3)
	SHA256_STEP(12, e, f, g, h, a, b, c, d, 0x72BE5D74)
	SHA256_STEP(13, d, e, f, g, h, a, b, c, 0x80DEB1FE)
	SHA256_STEP(14, c, d, e, f, g, h, a, b, 0x9BDC06A7)
	SHA256_STEP(15, b, c, d, e, f, g, h, a, 0xC19BF174)

	SHA256_STEP(16, a, b, c, d, e, f, g, h, 0xE49B69C1)
	SHA256_STEP(17, h, a, b, c, d, e, f, g, 0xEFBE4786)
	SHA256_STEP(18, g, h, a, b, c, d, e, f, 0x0FC19DC6)
	SHA256_STEP(19, f, g, h, a, b, c, d, e, 0x240CA1CC)
	SHA256_STEP(20, e, f, g, h, a, b, c, d, 0x2DE92C6F)
	SHA256_STEP(21, d, e, f, g, h, a, b, c, 0x4A7484AA)
	SHA256_STEP(22, c, d, e, f, g, h, a, b, 0x5CB0A9DC)
	SHA256_STEP(23, b, c, d, e, f, g, h, a, 0x76F988DA)

	SHA256_STEP(24, a, b, c, d, e, f, g, h, 0x983E5152)
	SHA256_STEP(25, h, a, b, c, d, e, f, g, 0xA831C66D)
	SHA256_STEP(26, g, h, a, b, c, d, e, f, 0xB00327C8)
	SHA256_STEP(27, f, g, h, a, b, c, d, e, 0xBF597FC7)
	SHA256_STEP(28, e, f, g, h, a, b, c, d, 0xC6E00BF3)
	SHA256_STEP(29, d, e, f, g, h, a, b, c, 0xD5A79147)
	SHA256_STEP(30, c, d, e, f, g, h, a, b, 0x06CA6351)
	SHA256_STEP(31, b, c, d, e, f, g, h, a, 0x14292967)

	SHA256_STEP(32, a, b, c, d, e, f, g, h, 0x27B70A85)
	SHA256_STEP(33, h, a, b, c, d, e, f, g, 0x2E1B2138)
	SHA256_STEP(34, g, h, a, b, c, d, e, f, 0x4D2C6DFC)
	SHA256_STEP(35, f, g, h, a, b, c, d, e, 0x53380D13)
	SHA256_STEP(36, e, f, g, h, a, b, c, d, 0x650A7354)
	SHA256_STEP(37, d, e, f, g, h, a, b, c, 0x766A0ABB)
	SHA256_STEP(38, c, d, e, f, g, h, a, b, 0x81C2C92E)
	SHA256_STEP(39, b, c, d, e, f, g, h, a, 0x92722C85)

	SHA256_STEP(40, a, b, c, d, e, f, g, h, 0xA2BFE8A1)
	SHA256_STEP(41, h, a, b, c, d, e, f, g, 0xA81A664B)
	SHA256_STEP(42, g, h, a, b, c, d, e, f, 0xC24B8B70)
	SHA256_STEP(43, f, g, h, a, b, c, d, e, 0xC76C51A3)
	SHA256_STEP(44, e, f, g, h, a, b, c, d, 0xD192E819)
	SHA256_STEP(45, d, e, f, g, h, a, b, c, 0xD6990624)
	SHA256_STEP(46, c, d, e, f, g, h, a, b, 0xF40E3585)
	SHA256_STEP(47, b, c, d, e, f, g, h, a, 0x106AA070)

	SHA256_STEP(48, a, b, c, d, e, f, g, h, 0x19A4C116)
	SHA256_STEP(49, h, a, b, c, d, e, f, g, 0x1E376C08)
	SHA256_STEP(50, g, h, a, b, c, d, e, f, 0x2748774C)
	SHA256_STEP(51, f, g, h, a, b, c, d, e, 0x34B0BCB5)
	SHA256_STEP(52, e, f, g, h, a, b, c, d, 0x391C0CB3)
	SHA256_STEP(53, d, e, f, g, h, a, b, c, 0x4ED8AA4A)
	SHA256_STEP(54, c, d, e, f, g, h, a, b, 0x5B9CCA4F)
	SHA256_STEP(55, b, c, d, e, f, g, h, a, 0x682E6FF3)

	SHA256_STEP(56, a, b, c, d, e, f, g, h, 0x748F82EE)
	SHA256_STEP(57, h, a, b, c, d, e, f, g, 0x78A5636F)
	SHA256_STEP(58, g, h, a, b, c, d, e, f, 0x84C87814)
	SHA256_STEP(59, f, g, h, a, b, c, d, e, 0x8CC70208)
	SHA256_STEP(60, e, f, g, h, a, b, c, d, 0x90BEFFFA)
	SHA256_STEP(61, d, e, f, g, h, a, b, c, 0xA4506CEB)
	SHA256_STEP(62, c, d, e, f, g, h, a, b, 0xBEF9A3F7)
	SHA256_STEP(63, b, c, d, e, f, g, h, a, 0xC67178F2)

	m_reg[0] += a;
	m_reg[1] += b;
	m_reg[2] += c;
	m_reg[3] += d;
	m_reg[4] += e;
	m_reg[5] += f;
	m_reg[6] += g;
	m_reg[7] += h;
	m_bytes += 64;
}

void Sha256_streambuf::reset() NOEXCEPT {
	setp(NULLPTR, NULLPTR);
	m_reg = SHA256_REG_INIT;
	m_bytes = 0;
}
Sha256_streambuf::int_type Sha256_streambuf::overflow(Sha256_streambuf::int_type c){
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

Sha256 Sha256_streambuf::finalize(){
	boost::uint64_t bytes = m_bytes;
	if(pptr()){
		bytes += static_cast<unsigned>(pptr() - m_chunk.begin());
	}
	sputc(traits_type::to_char_type(0x80));
	while(pptr() != m_chunk.begin() + 56){
		sputc(traits_type::to_char_type(0));
	}
	boost::uint64_t bits;
	store_be(bits, bytes * 8);
	xsputn(reinterpret_cast<const char *>(&bits), sizeof(bits));
	if(pptr() == m_chunk.end()){
		eat_chunk();
	}

	Sha256 sha256;
	for(unsigned i = 0; i < m_reg.size(); ++i){
		store_be(reinterpret_cast<boost::uint32_t *>(sha256.data())[i], m_reg[i]);
	}
	reset();
	return sha256;
}

Sha256_ostream::~Sha256_ostream(){ }

}
