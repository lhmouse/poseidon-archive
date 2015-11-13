// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_VINT50_HPP_
#define POSEIDON_VINT50_HPP_

#include "cxx_ver.hpp"
#include <cstddef>
#include <boost/cstdint.hpp>

namespace Poseidon {

// 最多输出七个字节，此函数返回后 write 指向最后一个写入的字节的后面。
template<typename OutputIterT>
void vuint50_to_binary(boost::uint64_t val, OutputIterT &write){
	for(unsigned i = 0; i < 6; ++i){
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
void vint50_to_binary(boost::int64_t val, OutputIterT &write){
	AUTO(encoded, static_cast<boost::uint64_t>(val));
	encoded <<= 1;
	if(val < 0){
		encoded = ~encoded;
	}
	encoded &= (1ull << 50) - 1;
	vuint50_to_binary(encoded, write);
}

// 返回值指向编码数据的结尾。成功返回 true，出错返回 false。
template<typename InputIterT>
bool vuint50_from_binary(boost::uint64_t &val, InputIterT &read, std::size_t count){
	val = 0;
	for(unsigned i = 0; i < 6; ++i){
		if(count == 0){
			return false;
		}
		const unsigned char by = *read;
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
	const unsigned char by = *read;
	++read;
	val |= static_cast<boost::uint64_t>(by) << 42;
	return true;
}
template<typename InputIterT>
bool vint50_from_binary(boost::int64_t &val, InputIterT &read, std::size_t count){
	boost::uint64_t encoded;
	const bool ret = vuint50_from_binary(encoded, read, count);
	if(ret){
		const bool negative = encoded & 1;
		encoded >>= 1;
		if(negative){
			encoded = ~encoded;
		}
		val = static_cast<boost::int64_t>(encoded);
	}
	return ret;
}

}

#endif
