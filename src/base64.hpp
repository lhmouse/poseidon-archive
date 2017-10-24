// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_BASE64_HPP_
#define POSEIDON_BASE64_HPP_

#include "cxx_util.hpp"
#include "stream_buffer.hpp"
#include <string>
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
	void put(char ch){
		put(&ch, 1);
	}
	void put(const char *str){
		put(str, std::strlen(str));
	}
	void put(const std::string &str){
		put(str.data(), str.size());
	}
	void put(const StreamBuffer &buffer);
	StreamBuffer finalize();
};

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
	void put(char ch){
		put(&ch, 1);
	}
	void put(const char *str){
		put(str, std::strlen(str));
	}
	void put(const std::string &str){
		put(str.data(), str.size());
	}
	void put(const StreamBuffer &buffer);
	StreamBuffer finalize();
};

extern std::string base64_encode(const void *data, std::size_t size);
extern std::string base64_encode(const char *str);
extern std::string base64_encode(const std::string &str);

extern std::string base64_decode(const void *data, std::size_t size);
extern std::string base64_decode(const char *str);
extern std::string base64_decode(const std::string &str);

}

#endif
