// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_VINT50_HPP_
#define POSEIDON_VINT50_HPP_

#include <cstddef>

namespace Poseidon {

// 最多输出七个字节，此函数返回后 write 指向最后一个写入的字节的后面。
template<typename OutputIterT>
void vuint50ToBinary(unsigned long long val, OutputIterT &write){
	for(unsigned i = 0; i < 6; ++i){
		const unsigned char by = val & 0x7F;
		val >>= 7;
		if(val == 0){
			*write = by;
			++write;
			return;
		}
		*write = (unsigned char)(by | 0x80);
		++write;
	}
	if(val != 0){
		*write = (unsigned char)val;
		++write;
	}
}
template<typename OutputIterT>
void vint50ToBinary(long long val, OutputIterT &write){
	unsigned long long encoded = val;
	encoded <<= 1;
	if(val < 0){
		encoded = ~encoded;
	}
	encoded &= (1ull << 50) - 1;
	vuint50ToBinary(encoded, write);
}

// 返回值指向编码数据的结尾。成功返回 true，出错返回 false。
template<typename InputIterT>
bool vuint50FromBinary(unsigned long long &val, InputIterT &read, std::size_t count){
	val = 0;
	for(unsigned i = 0; i < 6; ++i){
		if(count == 0){
			return false;
		}
		const unsigned char by = *read;
		++read;
		--count;
		val |= (unsigned long long)(by & 0x7F) << (i * 7);
		if(!(by & 0x80)){
			return true;
		}
	}
	if(count == 0){
		return false;
	}
	const unsigned char by = *read;
	++read;
	val |= (unsigned long long)by << 42;
	return true;
}
template<typename InputIterT>
bool vint50FromBinary(long long &val, InputIterT &read, std::size_t count){
	unsigned long long encoded;
	const bool ret = vuint50FromBinary(encoded, read, count);
	if(ret){
		const bool negative = encoded & 1;
		encoded >>= 1;
		if(negative){
			encoded = ~encoded;
		}
		val = (long long)encoded;
	}
	return ret;
}

}

#endif
