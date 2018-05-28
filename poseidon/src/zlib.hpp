// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ZLIB_HPP_
#define POSEIDON_ZLIB_HPP_

#include "stream_buffer.hpp"
#include <string>
#include <cstddef>
#include <cstring>
#include <boost/scoped_ptr.hpp>

#undef ZLIB_CONST
#define ZLIB_CONST 1
#include <zlib.h>

namespace Poseidon {

class Deflator {
private:
	::z_stream m_stream;
	Stream_buffer m_buffer;

public:
	explicit Deflator(bool gzip = false, int level = 8);
	~Deflator();

	Deflator(const Deflator &) = delete;
	Deflator &operator=(const Deflator &) = delete;

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
	void flush();
	Stream_buffer finalize();
};

class Inflator {
private:
	::z_stream m_stream;
	Stream_buffer m_buffer;

public:
	explicit Inflator(bool gzip = false);
	~Inflator();

	Inflator(const Inflator &) = delete;
	Inflator &operator=(const Inflator &) = delete;

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
	Stream_buffer finalize();
	void flush();
	void put(const Stream_buffer &buffer);
};

}

#endif
