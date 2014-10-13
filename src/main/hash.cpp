#include "../precompiled.hpp"
#include "hash.hpp"
#include <cstring>
#include <endian.h>
using namespace Poseidon;

namespace {

static const boost::uint32_t CRC32_TABLE[256] = {
	0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
	0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91,
	0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
	0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
	0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
	0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
	0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
	0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
	0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
	0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01,
	0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
	0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
	0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB,
	0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
	0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
	0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD,
	0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683,
	0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
	0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7,
	0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5,
	0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
	0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79,
	0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236, 0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F,
	0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
	0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713,
	0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21,
	0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
	0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
	0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB,
	0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
	0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD70693, 0x54DE5729, 0x23D967BF,
	0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94, 0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
};

inline boost::uint32_t rotl(boost::uint32_t u, int bits){
	return (u << bits) | (u >> (32 - bits));
}
inline boost::uint32_t rotr(boost::uint32_t u, int bits){
	return (u >> bits) | (u << (32 - bits));
}

void md5Chunk(boost::uint32_t (&result)[4], const unsigned char *chunk){
	// https://en.wikipedia.org/wiki/Md5
	const AUTO(w, (const boost::uint32_t *)chunk);

	register boost::uint32_t a = result[0];
	register boost::uint32_t b = result[1];
	register boost::uint32_t c = result[2];
	register boost::uint32_t d = result[3];

	register boost::uint32_t f, g;

#define MD5_STEP(i_, spec_, a_, b_, c_, d_, k_, r_)	\
	spec_(i_, a_, b_, c_, d_);	\
	a_ = b_ + rotl(a_ + f + k_ + htole32(w[g]), r_);

#define MD5_SPEC_0(i_, a_, b_, c_, d_)	(f = d_ ^ (b_ & (c_ ^ d_)), g = i_)
#define MD5_SPEC_1(i_, a_, b_, c_, d_)	(f = c_ ^ (d_ & (b_ ^ c_)), g = (5 * i_ + 1) % 16)
#define MD5_SPEC_2(i_, a_, b_, c_, d_)	(f = b_ ^ c_ ^ d_, g = (3 * i_ + 5) % 16)
#define MD5_SPEC_3(i_, a_, b_, c_, d_)	(f = c_ ^ (b_ | ~d_), g = (7 * i_) % 16)

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

	result[0] += a;
	result[1] += b;
	result[2] += c;
	result[3] += d;
}

void sha1Chunk(boost::uint32_t (&result)[5], const unsigned char *chunk){
	// https://en.wikipedia.org/wiki/Sha1
	boost::uint32_t w[80];

	for(std::size_t i = 0; i < 16; ++i){
		w[i] = htobe32(((const boost::uint32_t *)chunk)[i]);
	}
	for(std::size_t i = 16; i < 32; ++i){
		w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
	}
	for(std::size_t i = 32; i < 80; ++i){
		w[i] = rotl(w[i - 6] ^ w[i - 16] ^ w[i - 28] ^ w[i - 32], 2);
	}

	register boost::uint32_t a = result[0];
	register boost::uint32_t b = result[1];
	register boost::uint32_t c = result[2];
	register boost::uint32_t d = result[3];
	register boost::uint32_t e = result[4];

	register boost::uint32_t f, k;

#define SHA1_STEP(i_, spec_, a_, b_, c_, d_, e_)	\
	spec_(a_, b_, c_, d_, e_);	\
	e_ += rotl(a_, 5) + f + k + w[i_];	\
	b_ = rotl(b_, 30);

#define SHA1_SPEC_0(a_, b_, c_, d_, e_)	(f = d_ ^ (b_ & (c_ ^ d_)), k = 0x5A827999)
#define SHA1_SPEC_1(a_, b_, c_, d_, e_)	(f = b_ ^ c_ ^ d_, k = 0x6ED9EBA1)
#define SHA1_SPEC_2(a_, b_, c_, d_, e_)	(f = (b_ & (c_ | d_)) | (c_ & d_), k = 0x8F1BBCDC)
#define SHA1_SPEC_3(a_, b_, c_, d_, e_)	(f = b_ ^ c_ ^ d_, k = 0xCA62C1D6)

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

	result[0] += a;
	result[1] += b;
	result[2] += c;
	result[3] += d;
	result[4] += e;
}

}

namespace Poseidon {

boost::uint32_t crc32Sum(const void *data, std::size_t size){
	register boost::uint32_t reg = -1;
	AUTO(read, (const unsigned char *)data);
	for(std::size_t i = 0; i < size; ++i){
		reg = CRC32_TABLE[(reg ^ *read) & 0xFF] ^ (reg >> 8);
		++read;
	}
	return ~reg;
}

void md5Sum(unsigned char (&hash)[16], const void *data, std::size_t size){
	boost::uint32_t result[4] = {
		0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u
	};
	AUTO(read, (const unsigned char *)data);
	std::size_t remaining = size;
	while(remaining >= 64){
		md5Chunk(result, read);
		read += 64;
		remaining -= 64;
	}
	unsigned char chunk[64];
	std::memcpy(chunk, read, remaining);
	chunk[remaining] = 0x80;
	if(remaining >= 56){
		std::memset(chunk + remaining + 1, 0, 64 - remaining - 1);
		md5Chunk(result, chunk);
		std::memset(chunk, 0, 56);
	} else {
		std::memset(chunk + remaining + 1, 0, 56 - remaining - 1);
	}
	*(boost::uint64_t *)(chunk + 56) = htole64(size * 8ull);
	md5Chunk(result, chunk);
	for(unsigned i = 0; i < 4; ++i){
		((boost::uint32_t *)hash)[i] = htole32(result[i]);
	}
}

void sha1Sum(unsigned char (&hash)[20], const void *data, std::size_t size){
	boost::uint32_t result[5] = {
		0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u
	};
	AUTO(read, (const unsigned char *)data);
	std::size_t remaining = size;
	while(remaining >= 64){
		sha1Chunk(result, read);
		read += 64;
		remaining -= 64;
	}
	unsigned char chunk[64];
	std::memcpy(chunk, read, remaining);
	chunk[remaining] = 0x80;
	if(remaining >= 56){
		std::memset(chunk + remaining + 1, 0, 64 - remaining - 1);
		sha1Chunk(result, chunk);
		std::memset(chunk, 0, 56);
	} else {
		std::memset(chunk + remaining + 1, 0, 56 - remaining - 1);
	}
	*(boost::uint64_t *)(chunk + 56) = htobe64(size * 8ull);
	sha1Chunk(result, chunk);
	for(unsigned i = 0; i < 5; ++i){
		((boost::uint32_t *)hash)[i] = htobe32(result[i]);
	}
}

}
