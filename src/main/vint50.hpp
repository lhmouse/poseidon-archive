#ifndef POSEIDON_VINT50_HPP_
#define POSEIDON_VINT50_HPP_

namespace Poseidon {

// 最多输出七个字节，返回值是指向最后一个有效字节后面的指针。
inline unsigned char *vuint50ToBinary(unsigned long long val, unsigned char *write){
	for(unsigned i = 0; i < 7; ++i){
		*write = val & 0x7F;
		++write;
		val >>= 7;
		if(val == 0){
			return write;
		}
		write[-1] |= 0x80;
	}
	if(val != 0){
		*write = val;
		++write;
	}
	return write;
}
inline unsigned char *vint50ToBinary(long long val, unsigned char *write){
	unsigned long long encoded = val;
	encoded <<= 1;
	if(val < 0){
		encoded = ~encoded;
	}
	encoded &= (1ull << 50) - 1;
	return vuint50ToBinary(encoded, write);
}

// 返回值指向编码数据的结尾。如果解码出错返回空指针。
inline const unsigned char * vuint50FromBinary(unsigned long long &val,
	const unsigned char *read, const unsigned char *end)
{
	val = 0;
	for(unsigned i = 0; i < 7; ++i){
		if(read == end){
			return 0;
		}
		val |= (unsigned long long)(*read & 0x7F) << (i * 7);
		++read;
		if(!(read[-1] & 0x80)){
			return read;
		}
	}
	if(read == end){
		return 0;
	}
	val |= *read;
	++read;
	return read;
}
inline const unsigned char * vint50FromBinary(long long &val,
	const unsigned char *read, const unsigned char *end)
{
	unsigned long long encoded;
	const unsigned char *const ret = vuint50FromBinary(encoded, read, end);
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
