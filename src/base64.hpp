// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_BASE64_HPP_
#define POSEIDON_BASE64_HPP_

#include "cxx_util.hpp"
#include "stream_buffer.hpp"
#include <cstddef>
#include <cstring>

namespace Poseidon {

class Base64Encoder : NONCOPYABLE {
private:
	unsigned long m_seq;
	StreamBuffer m_buffer;

public:
	Base64Encoder();
	~Base64Encoder();

public:
	const StreamBuffer &get_buffer() const {
		return m_buffer;
	}
	StreamBuffer &get_buffer(){
		return m_buffer;
	}

	void clear();
	void put(const void *data, std::size_t size);
	void put(const StreamBuffer &buffer);
	StreamBuffer finalize();
};

inline std::string base64_encode(const void *data, std::size_t size){
	Base64Encoder enc;
	enc.put(data, size);
	return enc.finalize().dump_string();
}
inline std::string base64_encode(const char *str){
	Base64Encoder enc;
	enc.put(str, std::strlen(str));
	return enc.finalize().dump_string();
}
inline std::string base64_encode(const std::string &str){
	Base64Encoder enc;
	enc.put(str.data(), str.size());
	return enc.finalize().dump_string();
}
inline std::string base64_encode(const StreamBuffer &buffer){
	Base64Encoder enc;
	enc.put(buffer);
	return enc.finalize().dump_string();
}

class Base64Decoder : NONCOPYABLE {
private:
	unsigned long m_seq;
	StreamBuffer m_buffer;

public:
	Base64Decoder();
	~Base64Decoder();

public:
	const StreamBuffer &get_buffer() const {
		return m_buffer;
	}
	StreamBuffer &get_buffer(){
		return m_buffer;
	}

	void clear();
	void put(const void *data, std::size_t size);
	void put(const StreamBuffer &buffer);
	StreamBuffer finalize();
};

inline std::string base64_decode(const void *data, std::size_t size){
	Base64Decoder dec;
	dec.put(data, size);
	return dec.finalize().dump_string();
}
inline std::string base64_decode(const char *str){
	Base64Decoder dec;
	dec.put(str, std::strlen(str));
	return dec.finalize().dump_string();
}
inline std::string base64_decode(const std::string &str){
	Base64Decoder dec;
	dec.put(str.data(), str.size());
	return dec.finalize().dump_string();
}
inline std::string base64_decode(const StreamBuffer &buffer){
	Base64Decoder dec;
	dec.put(buffer);
	return dec.finalize().dump_string();
}

}

#endif
