// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HASH_HPP_
#define POSEIDON_HASH_HPP_

#include <cstddef>
#include <boost/cstdint.hpp>

namespace Poseidon {

extern boost::uint32_t crc32Sum(const void *data, std::size_t size);
extern void md5Sum(unsigned char (&hash)[16], const void *data, std::size_t size);
extern void sha1Sum(unsigned char (&hash)[20], const void *data, std::size_t size);
extern void sha256Sum(unsigned char (&hash)[32], const void *data, std::size_t size);

}

#endif
