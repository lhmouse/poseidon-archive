// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HASH_HPP_
#define POSEIDON_HASH_HPP_

#include <string>
#include <cstring>
#include <cstddef>
#include <boost/cstdint.hpp>
#include <boost/array.hpp>

namespace Poseidon {

typedef boost::uint32_t Crc32;

extern Crc32 crc32_hash(const void *data, std::size_t size);
inline Crc32 crc32_hash(const char *str){
	return crc32_hash(str, std::strlen(str));
}
inline Crc32 crc32_hash(const std::string &str){
	return crc32_hash(str.data(), str.size());
}

typedef boost::array<boost::uint8_t, 16> Md5;

extern Md5 md5_hash(const void *data, std::size_t size);
inline Md5 md5_hash(const char *str){
	return md5_hash(str, std::strlen(str));
}
inline Md5 md5_hash(const std::string &str){
	return md5_hash(str.data(), str.size());
}

typedef boost::array<boost::uint8_t, 20> Sha1;

extern Sha1 sha1_hash(const void *data, std::size_t size);
inline Sha1 sha1_hash(const char *str){
	return sha1_hash(str, std::strlen(str));
}
inline Sha1 sha1_hash(const std::string &str){
	return sha1_hash(str.data(), str.size());
}

typedef boost::array<boost::uint8_t, 32> Sha256;

extern Sha256 sha256_hash(const void *data, std::size_t size);
inline Sha256 sha256_hash(const char *str){
	return sha256_hash(str, std::strlen(str));
}
inline Sha256 sha256_hash(const std::string &str){
	return sha256_hash(str.data(), str.size());
}

}

#endif
