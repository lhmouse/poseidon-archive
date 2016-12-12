// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ZLIB_HPP_
#define POSEIDON_ZLIB_HPP_

#include "cxx_util.hpp"
#include "stream_buffer.hpp"
#include <boost/scoped_ptr.hpp>
#include <cstddef>

namespace Poseidon {

class Deflator : NONCOPYABLE {
private:
	struct Context;
	boost::scoped_ptr<Context> m_context;
	StreamBuffer m_buffer;

public:
	explicit Deflator(bool gzip = false, int level = 8);
	~Deflator();

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

class Inflator : NONCOPYABLE {
private:
	struct Context;
	boost::scoped_ptr<Context> m_context;
	StreamBuffer m_buffer;

public:
	explicit Inflator(bool gzip = false);
	~Inflator();

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

}

#endif
