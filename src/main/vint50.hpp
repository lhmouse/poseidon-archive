#ifndef POSEIDON_VINT50_HPP_
#define POSEIDON_VINT50_HPP_

namespace Poseidon {

// 最多输出七个字节，此函数返回后 write 指向最后一个写入的字节的后面。
template<typename OutputIter>
void vuint50ToBinary(unsigned long long val, OutputIter &write){
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
		*write = val;
		++write;
	}
}
template<typename OutputIter>
void vint50ToBinary(long long val, OutputIter &write){
	unsigned long long encoded = val;
	encoded <<= 1;
	if(val < 0){
		encoded = ~encoded;
	}
	encoded &= (1ull << 50) - 1;
	vuint50ToBinary(encoded, write);
}

// 返回值指向编码数据的结尾。成功返回 true，出错返回 false。
template<typename InputIter>
bool vuint50FromBinary(unsigned long long &val, InputIter &read, InputIter end){
	val = 0;
	for(unsigned i = 0; i < 6; ++i){
		if(read == end){
			return false;
		}
		const unsigned char by = *read;
		++read;
		val |= (unsigned long long)(by & 0x7F) << (i * 7);
		if(!(by & 0x80)){
			return true;
		}
	}
	if(read == end){
		return false;
	}
	const unsigned char by = *read;
	++read;
	val |= (unsigned long long)by << 42;
	return true;
}
template<typename InputIter>
bool vint50FromBinary(long long &val, InputIter &read, InputIter end){
	unsigned long long encoded;
	const bool ret = vuint50FromBinary(encoded, read, end);
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
