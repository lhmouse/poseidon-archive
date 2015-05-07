// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HASH_HPP_
#define POSEIDON_HASH_HPP_

#include <string>
#include <cstring>
#include <cstddef>
#include <boost/cstdint.hpp>

namespace Poseidon {

extern boost::uint32_t crc32Sum(const void *data, std::size_t size);
inline boost::uint32_t crc32Sum(const char *str){
	return crc32Sum(str, std::strlen(str));
}
inline boost::uint32_t crc32Sum(const std::string &str){
	return crc32Sum(str.data(), str.size());
}

extern void md5Sum(unsigned char (&hash)[16], const void *data, std::size_t size);
inline void md5Sum(unsigned char (&hash)[16], const char *str){
	md5Sum(hash, str, std::strlen(str));
}
inline void md5Sum(unsigned char (&hash)[16], const std::string &str){
	md5Sum(hash, str.data(), str.size());
}

extern void sha1Sum(unsigned char (&hash)[20], const void *data, std::size_t size);
inline void sha1Sum(unsigned char (&hash)[20], const char *str){
	sha1Sum(hash, str, std::strlen(str));
}
inline void sha1Sum(unsigned char (&hash)[20], const std::string &str){
	sha1Sum(hash, str.data(), str.size());
}

extern void sha256Sum(unsigned char (&hash)[32], const void *data, std::size_t size);
inline void sha256Sum(unsigned char (&hash)[32], const char *str){
	sha256Sum(hash, str, std::strlen(str));
}
inline void sha256Sum(unsigned char (&hash)[32], const std::string &str){
	sha256Sum(hash, str.data(), str.size());
}

}

#endif
