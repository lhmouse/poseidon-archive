// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ZLIB_STREAMS_HPP_
#define POSEIDON_ZLIB_STREAMS_HPP_

#include "cxx_ver.hpp"
#include <streambuf>
#include <istream>
#include <ostream>
#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include "stream_buffer.hpp"

namespace Poseidon {

struct ZlibContext;

class Zlib_streambuf : public std::streambuf {
private:
	StreamBuffer m_buffer;
	std::ios_base::openmode m_which;
	boost::shared_ptr<ZlibContext> m_get_ctx;
	boost::shared_ptr<ZlibContext> m_put_ctx;

public:
	explicit Zlib_streambuf(std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::streambuf()
		, m_buffer(), m_which(which)
	{
	}
	explicit Zlib_streambuf(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::streambuf()
		, m_buffer(STD_MOVE(buffer)), m_which(which)
	{
	}
	Zlib_streambuf(const Zlib_streambuf &rhs)
		: std::streambuf()
		, m_buffer(rhs.m_buffer), m_which(rhs.m_which)
	{
	}
	Zlib_streambuf &operator=(const Zlib_streambuf &rhs){
		sync();
		m_buffer = rhs.m_buffer;
		m_which = rhs.m_which;
		return *this;
	}
#ifdef POSEIDON_CXX11
	Zlib_streambuf(Zlib_streambuf &&rhs) noexcept
		: std::streambuf()
		, m_buffer(std::move(rhs.m_buffer)), m_which(rhs.m_which)
	{
	}
	Zlib_streambuf &operator=(Zlib_streambuf &&rhs) noexcept {
		sync();
		m_buffer = std::move(rhs.m_buffer);
		m_which = rhs.m_which;
		return *this;
	}
#endif
	~Zlib_streambuf() OVERRIDE;

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

	void swap(Zlib_streambuf &rhs) NOEXCEPT {
		sync();
		using std::swap;
		swap(m_buffer, rhs.m_buffer);
		swap(m_which, rhs.m_which);
	}
};

inline void swap(Zlib_streambuf &lhs, Zlib_streambuf &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

class Zlib_istream : public std::istream {
private:
	Zlib_streambuf m_sb;

public:
	explicit Zlib_istream(std::ios_base::openmode which = std::ios_base::in)
		: std::istream(&m_sb)
		, m_sb(which | std::ios_base::in)
	{
	}
	explicit Zlib_istream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in)
		: std::istream(&m_sb)
		, m_sb(STD_MOVE(buffer), which | std::ios_base::in)
	{
	}
	~Zlib_istream() OVERRIDE;

public:
	Zlib_streambuf *rdbuf() const {
		return const_cast<Zlib_streambuf *>(&m_sb);
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
	void swap(Zlib_istream &rhs) noexcept {
		std::istream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Zlib_istream &lhs, Zlib_istream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

class Zlib_ostream : public std::ostream {
private:
	Zlib_streambuf m_sb;

public:
	explicit Zlib_ostream(std::ios_base::openmode which = std::ios_base::out)
		: std::ostream(&m_sb)
		, m_sb(which | std::ios_base::out)
	{
	}
	explicit Zlib_ostream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::out)
		: std::ostream(&m_sb)
		, m_sb(STD_MOVE(buffer), which | std::ios_base::out)
	{
	}
	~Zlib_ostream() OVERRIDE;

public:
	Zlib_streambuf *rdbuf() const {
		return const_cast<Zlib_streambuf *>(&m_sb);
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
	void swap(Zlib_ostream &rhs) noexcept {
		std::ostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Zlib_ostream &lhs, Zlib_ostream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

class Zlib_stream : public std::iostream {
private:
	Zlib_streambuf m_sb;

public:
	explicit Zlib_stream(std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::iostream(&m_sb)
		, m_sb(which)
	{
	}
	explicit Zlib_stream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::iostream(&m_sb)
		, m_sb(STD_MOVE(buffer), which)
	{
	}
	~Zlib_stream() OVERRIDE;

public:
	Zlib_streambuf *rdbuf() const {
		return const_cast<Zlib_streambuf *>(&m_sb);
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
	void swap(Zlib_stream &rhs) noexcept {
		std::iostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Zlib_stream &lhs, Zlib_stream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

}

#endif
