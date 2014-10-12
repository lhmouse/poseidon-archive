// 这个文件是 MCF 的一部分。
// 有关具体授权说明，请参阅 MCFLicense.txt。
// Copyleft 2013 - 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_HASH_HPP_
#define POSEIDON_HTTP_HASH_HPP_

#include <cstddef>
#include <boost/cstdint.hpp>

namespace Poseidon {

extern boost::uint32_t crc32Sum(const void *data, std::size_t size);
extern void md5Sum(unsigned char (&hash)[16], const void *data, std::size_t size);
extern void sha1Sum(unsigned char (&hash)[20], const void *data, std::size_t size);

}

#endif
