// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_BUFFER_STREAMS_HPP_
#define POSEIDON_BUFFER_STREAMS_HPP_

#include "cxx_ver.hpp"
#include <streambuf>
#include <istream>
#include <ostream>
#include "stream_buffer.hpp"

namespace Poseidon {

class Buffer_streambuf : public std::streambuf {
private:
	StreamBuffer m_buffer;
	std::ios_base::openmode m_which;

public:
	explicit Buffer_streambuf(std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::streambuf()
		, m_buffer(), m_which(which)
	{
	}
	explicit Buffer_streambuf(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::streambuf()
		, m_buffer(STD_MOVE(buffer)), m_which(which)
	{
	}
#ifdef POSEIDON_CXX14
	Buffer_streambuf(Buffer_streambuf &&rhs) noexcept
		: std::streambuf(STD_MOVE(rhs))
		, m_buffer(STD_MOVE(rhs.buffer)), m_which(rhs.m_which)
	{
	}
	Buffer_streambuf &operator=(Buffer_streambuf &&rhs){
		sync();
		std::streambuf::operator=(std::move(rhs));
		m_buffer = std::move(rhs.m_buffer);
		m_which = rhs.m_which;
		return *this;
	}
#endif
	~Buffer_streambuf() OVERRIDE;

protected:
	void imbue(const std::locale &locale) OVERRIDE;

	Buffer_streambuf *setbuf(char_type *s, std::streamsize n) OVERRIDE;
	pos_type seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) OVERRIDE;
	pos_type seekpos(pos_type sp, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) OVERRIDE;
	int sync() OVERRIDE;

	std::streamsize showmanyc() OVERRIDE;
	std::streamsize xsgetn(char_type *s, std::streamsize n) OVERRIDE;
	int_type underflow() OVERRIDE;
	int_type uflow() OVERRIDE;

	int_type pbackfail(int_type c = traits_type::eof()) OVERRIDE;

	std::streamsize xsputn(const char_type *s, std::streamsize n) OVERRIDE;
	int_type overflow(int_type c = traits_type::eof()) OVERRIDE;

public:
	const StreamBuffer &get_buffer() const {
		return m_buffer;
	}
	StreamBuffer &get_buffer(){
		sync();
		return m_buffer;
	}
	void set_buffer(StreamBuffer buffer){
		sync();
		m_buffer.swap(buffer);
	}

#ifdef POSEIDON_CXX14
	void swap(Buffer_streambuf &rhs) noexcept {
		sync();
		using std::swap;
		swap(m_buffer, rhs.m_buffer);
		swap(m_which, rhs.m_which);
	}
#endif
};

class Buffer_istream : public std::istream {
protected:
	Buffer_streambuf m_buf;

public:
	explicit Buffer_istream(std::ios_base::openmode which = std::ios_base::in)
		: std::istream(&m_buf)
		, m_buf(which | std::ios_base::in)
	{
	}
	explicit Buffer_istream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in)
		: std::istream(&m_buf)
		, m_buf(STD_MOVE(buffer), which | std::ios_base::in)
	{
	}
	~Buffer_istream() OVERRIDE;

public:
	Buffer_streambuf *rdbuf() const {
		return const_cast<Buffer_streambuf *>(&m_buf);
	}

	const StreamBuffer &get_buffer() const {
		return m_buf.get_buffer();
	}
	StreamBuffer &get_buffer(){
		return m_buf.get_buffer();
	}
	void set_buffer(StreamBuffer buffer){
		m_buf.set_buffer(STD_MOVE(buffer));
	}

#ifdef POSEIDON_CXX14
	void swap(Buffer_istream &rhs) noexcept {
		std::istream::swap(rhs);
		using std::swap;
		swap(m_buf, rhs.m_buf);
	}
#endif
};
class Buffer_ostream : public std::ostream {
protected:
	Buffer_streambuf m_buf;

public:
	explicit Buffer_ostream(std::ios_base::openmode which = std::ios_base::out)
		: std::ostream(&m_buf)
		, m_buf(which | std::ios_base::out)
	{
	}
	explicit Buffer_ostream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::out)
		: std::ostream(&m_buf)
		, m_buf(STD_MOVE(buffer), which | std::ios_base::out)
	{
	}
	~Buffer_ostream() OVERRIDE;

public:
	Buffer_streambuf *rdbuf() const {
		return const_cast<Buffer_streambuf *>(&m_buf);
	}

	const StreamBuffer &get_buffer() const {
		return m_buf.get_buffer();
	}
	StreamBuffer &get_buffer(){
		return m_buf.get_buffer();
	}
	void set_buffer(StreamBuffer buffer){
		m_buf.set_buffer(STD_MOVE(buffer));
	}

#ifdef POSEIDON_CXX14
	void swap(Buffer_ostream &rhs) noexcept {
		std::ostream::swap(rhs);
		using std::swap;
		swap(m_buf, rhs.m_buf);
	}
#endif
};
class Buffer_iostream : public std::iostream {
protected:
	Buffer_streambuf m_buf;

public:
	explicit Buffer_iostream(std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::iostream(&m_buf)
		, m_buf(which)
	{
	}
	explicit Buffer_iostream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::iostream(&m_buf)
		, m_buf(STD_MOVE(buffer), which)
	{
	}
	~Buffer_iostream() OVERRIDE;

public:
	Buffer_streambuf *rdbuf() const {
		return const_cast<Buffer_streambuf *>(&m_buf);
	}

	const StreamBuffer &get_buffer() const {
		return m_buf.get_buffer();
	}
	StreamBuffer &get_buffer(){
		return m_buf.get_buffer();
	}
	void set_buffer(StreamBuffer buffer){
		m_buf.set_buffer(STD_MOVE(buffer));
	}

#ifdef POSEIDON_CXX14
	void swap(Buffer_ostream &rhs) noexcept {
		std::ostream::swap(rhs);
		using std::swap;
		swap(m_buf, rhs.m_buf);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Buffer_streambuf &lhs, Buffer_streambuf &rhs) noexcept {
	lhs.swap(rhs);
}

inline void swap(Buffer_istream &lhs, Buffer_istream &rhs) noexcept {
	lhs.swap(rhs);
}
inline void swap(Buffer_ostream &lhs, Buffer_ostream &rhs) noexcept {
	lhs.swap(rhs);
}
inline void swap(Buffer_iostream &lhs, Buffer_iostream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

}

#endif
