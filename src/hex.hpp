// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HEX_HPP_
#define POSEIDON_HEX_HPP_

#include "cxx_util.hpp"
#include "stream_buffer.hpp"
#include <cstddef>
#include <cstring>

namespace Poseidon {

class HexEncoder : NONCOPYABLE {
private:
	bool m_upper_case;
	StreamBuffer m_buffer;

public:
	explicit HexEncoder(bool upper_case = false);
	~HexEncoder();

public:
	bool is_upper_case() const {
		return m_upper_case;
	}
	void set_upper_case(bool upper_case){
		m_upper_case = upper_case;
	}

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

inline std::string hex_encode(const void *data, std::size_t size, bool upper_case = false){
	HexEncoder enc;
	enc.set_upper_case(upper_case);
	enc.put(data, size);
	return enc.finalize().dump_string();
}
inline std::string hex_encode(const char *str, bool upper_case = false){
	HexEncoder enc;
	enc.set_upper_case(upper_case);
	enc.put(str, std::strlen(str));
	return enc.finalize().dump_string();
}
inline std::string hex_encode(const std::string &str, bool upper_case = false){
	HexEncoder enc;
	enc.set_upper_case(upper_case);
	enc.put(str.data(), str.size());
	return enc.finalize().dump_string();
}
inline std::string hex_encode(const StreamBuffer &buffer, bool upper_case = false){
	HexEncoder enc;
	enc.set_upper_case(upper_case);
	enc.put(buffer);
	return enc.finalize().dump_string();
}

class HexDecoder : NONCOPYABLE {
private:
	unsigned m_seq;
	StreamBuffer m_buffer;

public:
	HexDecoder();
	~HexDecoder();

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

inline std::string hex_decode(const void *data, std::size_t size){
	HexDecoder dec;
	dec.put(data, size);
	return dec.finalize().dump_string();
}
inline std::string hex_decode(const char *str){
	HexDecoder dec;
	dec.put(str, std::strlen(str));
	return dec.finalize().dump_string();
}
inline std::string hex_decode(const std::string &str){
	HexDecoder dec;
	dec.put(str.data(), str.size());
	return dec.finalize().dump_string();
}
inline std::string hex_decode(const StreamBuffer &buffer){
	HexDecoder dec;
	dec.put(buffer);
	return dec.finalize().dump_string();
}

}

#endif
