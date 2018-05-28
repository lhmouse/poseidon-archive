// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_BASE64_HPP_
#define POSEIDON_BASE64_HPP_

#include "stream_buffer.hpp"
#include <string>
#include <cstddef>
#include <cstring>

namespace Poseidon {

class Base64_encoder {
private:
	unsigned long m_seq;
	Stream_buffer m_buffer;

public:
	Base64_encoder();
	~Base64_encoder();

	Base64_encoder(const Base64_encoder &);
	Base64_encoder &operator=(const Base64_encoder &);

public:
	const Stream_buffer & get_buffer() const {
		return m_buffer;
	}
	Stream_buffer & get_buffer(){
		return m_buffer;
	}

	void clear();
	void put(const void *data, std::size_t size);
	void put(char ch){
		put(&ch, 1);
	}
	void put(int ch){
		put(static_cast<char>(ch));
	}
	void put(const char *str){
		put(str, std::strlen(str));
	}
	void put(const std::string &str){
		put(str.data(), str.size());
	}
	void put(const Stream_buffer &buffer);
	Stream_buffer finalize();
};

class Base64_decoder {
private:
	unsigned long m_seq;
	Stream_buffer m_buffer;

public:
	Base64_decoder();
	~Base64_decoder();

	Base64_decoder(const Base64_decoder &);
	Base64_decoder &operator=(const Base64_decoder &);

public:
	const Stream_buffer & get_buffer() const {
		return m_buffer;
	}
	Stream_buffer & get_buffer(){
		return m_buffer;
	}

	void clear();
	void put(const void *data, std::size_t size);
	void put(char ch){
		put(&ch, 1);
	}
	void put(int ch){
		put(static_cast<char>(ch));
	}
	void put(const char *str){
		put(str, std::strlen(str));
	}
	void put(const std::string &str){
		put(str.data(), str.size());
	}
	void put(const Stream_buffer &buffer);
	Stream_buffer finalize();
};

extern std::string base64_encode(const void *data, std::size_t size);
extern std::string base64_encode(const char *str);
extern std::string base64_encode(const std::string &str);

extern std::string base64_decode(const void *data, std::size_t size);
extern std::string base64_decode(const char *str);
extern std::string base64_decode(const std::string &str);

}

#endif
