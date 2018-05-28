// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HEX_HPP_
#define POSEIDON_HEX_HPP_

#include "stream_buffer.hpp"
#include <string>
#include <cstddef>
#include <cstring>

namespace Poseidon {

class Hex_encoder {
private:
	bool m_upper_case;
	Stream_buffer m_buffer;

public:
	explicit Hex_encoder(bool upper_case = false);
	~Hex_encoder();

	Hex_encoder(const Hex_encoder &) = delete;
	Hex_encoder &operator=(const Hex_encoder &) = delete;

public:
	bool is_upper_case() const {
		return m_upper_case;
	}
	void set_upper_case(bool upper_case){
		m_upper_case = upper_case;
	}

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

class Hex_decoder {
private:
	unsigned m_seq;
	Stream_buffer m_buffer;

public:
	Hex_decoder();
	~Hex_decoder();

	Hex_decoder(const Hex_decoder &) = delete;
	Hex_decoder &operator=(const Hex_decoder &) = delete;

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

extern std::string hex_encode(const void *data, std::size_t size, bool upper_case = false);
extern std::string hex_encode(const char *str, bool upper_case = false);
extern std::string hex_encode(const std::string &str, bool upper_case = false);

extern std::string hex_decode(const void *data, std::size_t size);
extern std::string hex_decode(const char *str);
extern std::string hex_decode(const std::string &str);

}

#endif
