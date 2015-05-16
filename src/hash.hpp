// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HASH_HPP_
#define POSEIDON_HASH_HPP_

#include <string>
#include <cstring>
#include <cstddef>
#include <boost/cstdint.hpp>
#include <boost/array.hpp>

namespace Poseidon {

typedef boost::uint32_t Crc32;

extern Crc32 crc32Hash(const void *data, std::size_t size);
inline Crc32 crc32Hash(const char *str){
	return crc32Hash(str, std::strlen(str));
}
inline Crc32 crc32Hash(const std::string &str){
	return crc32Hash(str.data(), str.size());
}

typedef boost::array<boost::uint8_t, 16> Md5;

extern Md5 md5Hash(const void *data, std::size_t size);
inline Md5 md5Hash(const char *str){
	return md5Hash(str, std::strlen(str));
}
inline Md5 md5Hash(const std::string &str){
	return md5Hash(str.data(), str.size());
}

typedef boost::array<boost::uint8_t, 20> Sha1;

extern Sha1 sha1Hash(const void *data, std::size_t size);
inline Sha1 sha1Hash(const char *str){
	return sha1Hash(str, std::strlen(str));
}
inline Sha1 sha1Hash(const std::string &str){
	return sha1Hash(str.data(), str.size());
}

typedef boost::array<boost::uint8_t, 32> Sha256;

extern Sha256 sha256Hash(const void *data, std::size_t size);
inline Sha256 sha256Hash(const char *str){
	return sha256Hash(str, std::strlen(str));
}
inline Sha256 sha256Hash(const std::string &str){
	return sha256Hash(str.data(), str.size());
}

}

#endif
