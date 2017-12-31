// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_BUFFER_STREAMS_HPP_
#define POSEIDON_BUFFER_STREAMS_HPP_

#include "cxx_ver.hpp"
#include <streambuf>
#include <istream>
#include <ostream>
#include <boost/array.hpp>
#include "stream_buffer.hpp"

namespace Poseidon {

class Buffer_streambuf : public std::streambuf {
private:
	StreamBuffer m_buffer;
	std::ios_base::openmode m_which;
	boost::array<char, 32> m_get_area;

public:
	explicit Buffer_streambuf(std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::streambuf()
		, m_buffer(), m_which(which)
	{ }
	explicit Buffer_streambuf(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::streambuf()
		, m_buffer(STD_MOVE(buffer)), m_which(which)
	{ }
#ifdef POSEIDON_CXX11
	Buffer_streambuf(Buffer_streambuf &&rhs) noexcept
		: std::streambuf()
		, m_buffer(std::move(rhs.get_buffer())), m_which(rhs.m_which)
	{ }
	Buffer_streambuf &operator=(Buffer_streambuf &&rhs) noexcept {
		sync();
		m_buffer = std::move(rhs.get_buffer());
		m_which = rhs.m_which;
		return *this;
	}
#endif
	~Buffer_streambuf() OVERRIDE;

protected:
	int sync() OVERRIDE;

	std::streamsize showmanyc() OVERRIDE;
	std::streamsize xsgetn(char_type *s, std::streamsize n) OVERRIDE;
	int_type underflow() OVERRIDE;

	int_type pbackfail(int_type c = traits_type::eof()) OVERRIDE;

	std::streamsize xsputn(const char_type *s, std::streamsize n) OVERRIDE;
	int_type overflow(int_type c = traits_type::eof()) OVERRIDE;

public:
	StreamBuffer &get_buffer(){
		sync();
		return m_buffer;
	}
	void set_buffer(StreamBuffer buffer){
		sync();
		m_buffer.swap(buffer);
	}

	void swap(Buffer_streambuf &rhs) NOEXCEPT {
		sync();
		using std::swap;
		swap(m_buffer, rhs.get_buffer());
		swap(m_which, rhs.m_which);
	}
};

inline void swap(Buffer_streambuf &lhs, Buffer_streambuf &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

class Buffer_istream : public std::istream {
private:
	Buffer_streambuf m_sb;

public:
	explicit Buffer_istream(std::ios_base::openmode which = std::ios_base::in)
		: std::istream(&m_sb)
		, m_sb(which | std::ios_base::in)
	{ }
	explicit Buffer_istream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in)
		: std::istream(&m_sb)
		, m_sb(STD_MOVE(buffer), which | std::ios_base::in)
	{ }
	~Buffer_istream() OVERRIDE;

public:
	Buffer_streambuf *rdbuf() const {
		return const_cast<Buffer_streambuf *>(&m_sb);
	}

	const StreamBuffer &get_buffer() const {
		return rdbuf()->get_buffer();
	}
	StreamBuffer &get_buffer(){
		return rdbuf()->get_buffer();
	}
	void set_buffer(StreamBuffer buffer){
		rdbuf()->set_buffer(STD_MOVE(buffer));
	}

#ifdef POSEIDON_CXX14
	void swap(Buffer_istream &rhs) noexcept {
		std::istream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Buffer_istream &lhs, Buffer_istream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

class Buffer_ostream : public std::ostream {
private:
	Buffer_streambuf m_sb;

public:
	explicit Buffer_ostream(std::ios_base::openmode which = std::ios_base::out)
		: std::ostream(&m_sb)
		, m_sb(which | std::ios_base::out)
	{ }
	explicit Buffer_ostream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::out)
		: std::ostream(&m_sb)
		, m_sb(STD_MOVE(buffer), which | std::ios_base::out)
	{ }
	~Buffer_ostream() OVERRIDE;

public:
	Buffer_streambuf *rdbuf() const {
		return const_cast<Buffer_streambuf *>(&m_sb);
	}

	const StreamBuffer &get_buffer() const {
		return rdbuf()->get_buffer();
	}
	StreamBuffer &get_buffer(){
		return rdbuf()->get_buffer();
	}
	void set_buffer(StreamBuffer buffer){
		rdbuf()->set_buffer(STD_MOVE(buffer));
	}

#ifdef POSEIDON_CXX14
	void swap(Buffer_ostream &rhs) noexcept {
		std::ostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Buffer_ostream &lhs, Buffer_ostream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

class Buffer_stream : public std::iostream {
private:
	Buffer_streambuf m_sb;

public:
	explicit Buffer_stream(std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::iostream(&m_sb)
		, m_sb(which)
	{ }
	explicit Buffer_stream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::iostream(&m_sb)
		, m_sb(STD_MOVE(buffer), which)
	{ }
	~Buffer_stream() OVERRIDE;

public:
	Buffer_streambuf *rdbuf() const {
		return const_cast<Buffer_streambuf *>(&m_sb);
	}

	const StreamBuffer &get_buffer() const {
		return rdbuf()->get_buffer();
	}
	StreamBuffer &get_buffer(){
		return rdbuf()->get_buffer();
	}
	void set_buffer(StreamBuffer buffer){
		rdbuf()->set_buffer(STD_MOVE(buffer));
	}

#ifdef POSEIDON_CXX14
	void swap(Buffer_stream &rhs) noexcept {
		std::iostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Buffer_stream &lhs, Buffer_stream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

}

#endif
