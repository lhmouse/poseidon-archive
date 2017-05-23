// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_VINT64_HPP_
#define POSEIDON_VINT64_HPP_

#include <cstddef>
#include <boost/cstdint.hpp>

namespace Poseidon {

// 最多输出七个字节，此函数返回后 write 指向最后一个写入的字节的后面。
template<typename OutputIterT>
void vuint64_to_binary(boost::uint64_t val, OutputIterT &write){
	for(unsigned i = 0; i < 8; ++i){
		const unsigned char by = val & 0x7F;
		val >>= 7;
		if(val == 0){
			*write = by;
			++write;
			return;
		}
		*write = static_cast<unsigned char>(by | 0x80);
		++write;
	}
	if(val != 0){
		*write = static_cast<unsigned char>(val);
		++write;
	}
}
template<typename OutputIterT>
void vint64_to_binary(boost::int64_t val, OutputIterT &write){
	boost::uint64_t encoded = static_cast<boost::uint64_t>(val);
	encoded = (encoded << 1) ^ -(encoded >> 63);
	vuint64_to_binary(encoded, write);
}

// 返回值指向编码数据的结尾。成功返回 true，出错返回 false。
template<typename InputIterT>
bool vuint64_from_binary(boost::uint64_t &val, InputIterT &read, std::size_t count){
	val = 0;
	for(unsigned i = 0; i < 8; ++i){
		if(count == 0){
			return false;
		}
		const int by = (sizeof(*read) == 1) ? (*read & 0xFF) : *read;
		if(static_cast<unsigned>(by) > 0xFF){
			return false;
		}
		++read;
		--count;
		val |= static_cast<boost::uint64_t>(by & 0x7F) << (i * 7);
		if(!(by & 0x80)){
			return true;
		}
	}
	if(count == 0){
		return false;
	}
	const int by = (sizeof(*read) == 1) ? (*read & 0xFF) : *read;
	if(static_cast<unsigned>(by) > 0xFF){
		return false;
	}
	++read;
	val |= static_cast<boost::uint64_t>(by) << (8 * 7);
	return true;
}
template<typename InputIterT>
bool vint64_from_binary(boost::int64_t &val, InputIterT &read, std::size_t count){
	boost::uint64_t encoded;
	const bool ret = vuint64_from_binary(encoded, read, count);
	if(ret){
		encoded = (encoded >> 1) ^ -(encoded & 1);
		val = static_cast<boost::int64_t>(encoded);
	}
	return ret;
}

}

#endif
