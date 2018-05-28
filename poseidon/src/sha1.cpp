// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "sha1.hpp"
#include "endian.hpp"
#include <x86intrin.h>

namespace Poseidon {

namespace {
	CONSTEXPR const boost::array<std::uint32_t, 5> g_sha1_reg_init = {{ 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u }};
}

Sha1_streambuf::Sha1_streambuf()
	: m_reg(g_sha1_reg_init), m_bytes(0)
{
	//
}
Sha1_streambuf::~Sha1_streambuf(){
	//
}

void Sha1_streambuf::eat_chunk(){
	// https://en.wikipedia.org/wiki/SHA-1
	boost::array<std::uint32_t, 80> w;
	for(std::size_t i = 0; i < 16; ++i){
		w[i] = load_be(reinterpret_cast<const std::uint32_t *>(m_chunk.data())[i]);
	}
	for(std::size_t i = 16; i < 32; ++i){
		w[i] = __rold(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
	}
	for(std::size_t i = 32; i < 80; ++i){
		w[i] = __rold(w[i - 6] ^ w[i - 16] ^ w[i - 28] ^ w[i - 32], 2);
	}

	std::uint32_t a = m_reg[0];
	std::uint32_t b = m_reg[1];
	std::uint32_t c = m_reg[2];
	std::uint32_t d = m_reg[3];
	std::uint32_t e = m_reg[4];

	std::uint32_t f, k;

#define SHA1_STEP(i_, spec_, a_, b_, c_, d_, e_)	\
	spec_(a_, b_, c_, d_, e_);	\
	e_ += __rold(a_, 5) + f + k + w[i_];	\
	b_ = __rold(b_, 30);

#define SHA1_SPEC_0(a_, b_, c_, d_, e_) (f = d_ ^ (b_ & (c_ ^ d_)), k = 0x5A827999)
#define SHA1_SPEC_1(a_, b_, c_, d_, e_) (f = b_ ^ c_ ^ d_, k = 0x6ED9EBA1)
#define SHA1_SPEC_2(a_, b_, c_, d_, e_) (f = (b_ & (c_ | d_)) | (c_ & d_), k = 0x8F1BBCDC)
#define SHA1_SPEC_3(a_, b_, c_, d_, e_) (f = b_ ^ c_ ^ d_, k = 0xCA62C1D6)

	SHA1_STEP( 0, SHA1_SPEC_0, a, b, c, d, e)
	SHA1_STEP( 1, SHA1_SPEC_0, e, a, b, c, d)
	SHA1_STEP( 2, SHA1_SPEC_0, d, e, a, b, c)
	SHA1_STEP( 3, SHA1_SPEC_0, c, d, e, a, b)
	SHA1_STEP( 4, SHA1_SPEC_0, b, c, d, e, a)
	SHA1_STEP( 5, SHA1_SPEC_0, a, b, c, d, e)
	SHA1_STEP( 6, SHA1_SPEC_0, e, a, b, c, d)
	SHA1_STEP( 7, SHA1_SPEC_0, d, e, a, b, c)
	SHA1_STEP( 8, SHA1_SPEC_0, c, d, e, a, b)
	SHA1_STEP( 9, SHA1_SPEC_0, b, c, d, e, a)
	SHA1_STEP(10, SHA1_SPEC_0, a, b, c, d, e)
	SHA1_STEP(11, SHA1_SPEC_0, e, a, b, c, d)
	SHA1_STEP(12, SHA1_SPEC_0, d, e, a, b, c)
	SHA1_STEP(13, SHA1_SPEC_0, c, d, e, a, b)
	SHA1_STEP(14, SHA1_SPEC_0, b, c, d, e, a)
	SHA1_STEP(15, SHA1_SPEC_0, a, b, c, d, e)
	SHA1_STEP(16, SHA1_SPEC_0, e, a, b, c, d)
	SHA1_STEP(17, SHA1_SPEC_0, d, e, a, b, c)
	SHA1_STEP(18, SHA1_SPEC_0, c, d, e, a, b)
	SHA1_STEP(19, SHA1_SPEC_0, b, c, d, e, a)

	SHA1_STEP(20, SHA1_SPEC_1, a, b, c, d, e)
	SHA1_STEP(21, SHA1_SPEC_1, e, a, b, c, d)
	SHA1_STEP(22, SHA1_SPEC_1, d, e, a, b, c)
	SHA1_STEP(23, SHA1_SPEC_1, c, d, e, a, b)
	SHA1_STEP(24, SHA1_SPEC_1, b, c, d, e, a)
	SHA1_STEP(25, SHA1_SPEC_1, a, b, c, d, e)
	SHA1_STEP(26, SHA1_SPEC_1, e, a, b, c, d)
	SHA1_STEP(27, SHA1_SPEC_1, d, e, a, b, c)
	SHA1_STEP(28, SHA1_SPEC_1, c, d, e, a, b)
	SHA1_STEP(29, SHA1_SPEC_1, b, c, d, e, a)
	SHA1_STEP(30, SHA1_SPEC_1, a, b, c, d, e)
	SHA1_STEP(31, SHA1_SPEC_1, e, a, b, c, d)
	SHA1_STEP(32, SHA1_SPEC_1, d, e, a, b, c)
	SHA1_STEP(33, SHA1_SPEC_1, c, d, e, a, b)
	SHA1_STEP(34, SHA1_SPEC_1, b, c, d, e, a)
	SHA1_STEP(35, SHA1_SPEC_1, a, b, c, d, e)
	SHA1_STEP(36, SHA1_SPEC_1, e, a, b, c, d)
	SHA1_STEP(37, SHA1_SPEC_1, d, e, a, b, c)
	SHA1_STEP(38, SHA1_SPEC_1, c, d, e, a, b)
	SHA1_STEP(39, SHA1_SPEC_1, b, c, d, e, a)

	SHA1_STEP(40, SHA1_SPEC_2, a, b, c, d, e)
	SHA1_STEP(41, SHA1_SPEC_2, e, a, b, c, d)
	SHA1_STEP(42, SHA1_SPEC_2, d, e, a, b, c)
	SHA1_STEP(43, SHA1_SPEC_2, c, d, e, a, b)
	SHA1_STEP(44, SHA1_SPEC_2, b, c, d, e, a)
	SHA1_STEP(45, SHA1_SPEC_2, a, b, c, d, e)
	SHA1_STEP(46, SHA1_SPEC_2, e, a, b, c, d)
	SHA1_STEP(47, SHA1_SPEC_2, d, e, a, b, c)
	SHA1_STEP(48, SHA1_SPEC_2, c, d, e, a, b)
	SHA1_STEP(49, SHA1_SPEC_2, b, c, d, e, a)
	SHA1_STEP(50, SHA1_SPEC_2, a, b, c, d, e)
	SHA1_STEP(51, SHA1_SPEC_2, e, a, b, c, d)
	SHA1_STEP(52, SHA1_SPEC_2, d, e, a, b, c)
	SHA1_STEP(53, SHA1_SPEC_2, c, d, e, a, b)
	SHA1_STEP(54, SHA1_SPEC_2, b, c, d, e, a)
	SHA1_STEP(55, SHA1_SPEC_2, a, b, c, d, e)
	SHA1_STEP(56, SHA1_SPEC_2, e, a, b, c, d)
	SHA1_STEP(57, SHA1_SPEC_2, d, e, a, b, c)
	SHA1_STEP(58, SHA1_SPEC_2, c, d, e, a, b)
	SHA1_STEP(59, SHA1_SPEC_2, b, c, d, e, a)

	SHA1_STEP(60, SHA1_SPEC_3, a, b, c, d, e)
	SHA1_STEP(61, SHA1_SPEC_3, e, a, b, c, d)
	SHA1_STEP(62, SHA1_SPEC_3, d, e, a, b, c)
	SHA1_STEP(63, SHA1_SPEC_3, c, d, e, a, b)
	SHA1_STEP(64, SHA1_SPEC_3, b, c, d, e, a)
	SHA1_STEP(65, SHA1_SPEC_3, a, b, c, d, e)
	SHA1_STEP(66, SHA1_SPEC_3, e, a, b, c, d)
	SHA1_STEP(67, SHA1_SPEC_3, d, e, a, b, c)
	SHA1_STEP(68, SHA1_SPEC_3, c, d, e, a, b)
	SHA1_STEP(69, SHA1_SPEC_3, b, c, d, e, a)
	SHA1_STEP(70, SHA1_SPEC_3, a, b, c, d, e)
	SHA1_STEP(71, SHA1_SPEC_3, e, a, b, c, d)
	SHA1_STEP(72, SHA1_SPEC_3, d, e, a, b, c)
	SHA1_STEP(73, SHA1_SPEC_3, c, d, e, a, b)
	SHA1_STEP(74, SHA1_SPEC_3, b, c, d, e, a)
	SHA1_STEP(75, SHA1_SPEC_3, a, b, c, d, e)
	SHA1_STEP(76, SHA1_SPEC_3, e, a, b, c, d)
	SHA1_STEP(77, SHA1_SPEC_3, d, e, a, b, c)
	SHA1_STEP(78, SHA1_SPEC_3, c, d, e, a, b)
	SHA1_STEP(79, SHA1_SPEC_3, b, c, d, e, a)

	m_reg[0] += a;
	m_reg[1] += b;
	m_reg[2] += c;
	m_reg[3] += d;
	m_reg[4] += e;
	m_bytes += 64;
}

void Sha1_streambuf::reset() NOEXCEPT {
	setp(NULLPTR, NULLPTR);
	m_reg = g_sha1_reg_init;
	m_bytes = 0;
}
Sha1_streambuf::int_type Sha1_streambuf::overflow(Sha1_streambuf::int_type c){
	if(pptr() == m_chunk.end()){
		eat_chunk();
		setp(NULLPTR, NULLPTR);
	}
	if(traits_type::eq_int_type(c, traits_type::eof())){
		return traits_type::not_eof(c);
	}
	setp(m_chunk.begin(), m_chunk.end());
	*(pptr()) = traits_type::to_char_type(c);
	pbump(1);
	return c;
}

Sha1 Sha1_streambuf::finalize(){
	std::uint64_t bytes = m_bytes;
	if(pptr()){
		bytes += static_cast<unsigned>(pptr() - m_chunk.begin());
	}
	static const unsigned char s_terminator[65] = { 0x80 };
	xsputn(reinterpret_cast<const char *>(s_terminator), static_cast<int>(64 - (bytes + 8) % 64));
	std::uint64_t bits;
	store_be(bits, bytes * 8);
	xsputn(reinterpret_cast<const char *>(&bits), sizeof(bits));
	if(pptr() == m_chunk.end()){
		eat_chunk();
	}

	Sha1 sha1;
	for(unsigned i = 0; i < m_reg.size(); ++i){
		store_be(reinterpret_cast<std::uint32_t *>(sha1.data())[i], m_reg[i]);
	}
	reset();
	return sha1;
}

Sha1_ostream::~Sha1_ostream(){
	//
}

}
