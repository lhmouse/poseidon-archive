// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_BASE64_HPP_
#define POSEIDON_BASE64_HPP_

#include "cxx_ver.hpp"
#include <streambuf>
#include <istream>
#include <ostream>
#include <boost/array.hpp>
#include "stream_buffer.hpp"

namespace Poseidon {

class Base64_streambuf : public std::streambuf {
private:
	StreamBuffer m_buffer;
	std::ios_base::openmode m_which;
	boost::array<char, 3> m_get_area;
	boost::array<char, 3> m_put_area;

public:
	explicit Base64_streambuf(std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::streambuf()
		, m_buffer(), m_which(which)
	{
	}
	explicit Base64_streambuf(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::streambuf()
		, m_buffer(STD_MOVE(buffer)), m_which(which)
	{
	}
	Base64_streambuf(const Base64_streambuf &rhs)
		: std::streambuf()
		, m_buffer(rhs.m_buffer), m_which(rhs.m_which)
	{
	}
	Base64_streambuf &operator=(const Base64_streambuf &rhs){
		sync();
		m_buffer = rhs.m_buffer;
		m_which = rhs.m_which;
		return *this;
	}
#ifdef POSEIDON_CXX11
	Base64_streambuf(Base64_streambuf &&rhs) noexcept
		: std::streambuf()
		, m_buffer(std::move(rhs.m_buffer)), m_which(rhs.m_which)
	{
	}
	Base64_streambuf &operator=(Base64_streambuf &&rhs) noexcept {
		sync();
		m_buffer = std::move(rhs.m_buffer);
		m_which = rhs.m_which;
		return *this;
	}
#endif
	~Base64_streambuf() OVERRIDE;

protected:
	int sync() OVERRIDE;

	int_type underflow() OVERRIDE;

	int_type pbackfail(int_type c = traits_type::eof()) OVERRIDE;

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

	void swap(Base64_streambuf &rhs) NOEXCEPT {
		sync();
		using std::swap;
		swap(m_buffer, rhs.m_buffer);
		swap(m_which, rhs.m_which);
	}
};

inline void swap(Base64_streambuf &lhs, Base64_streambuf &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

class Base64_istream : public std::istream {
private:
	Base64_streambuf m_sb;

public:
	explicit Base64_istream(std::ios_base::openmode which = std::ios_base::in)
		: std::istream(&m_sb)
		, m_sb(which | std::ios_base::in)
	{
	}
	explicit Base64_istream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in)
		: std::istream(&m_sb)
		, m_sb(STD_MOVE(buffer), which | std::ios_base::in)
	{
	}
	~Base64_istream() OVERRIDE;

public:
	Base64_streambuf *rdbuf() const {
		return const_cast<Base64_streambuf *>(&m_sb);
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
	void swap(Base64_istream &rhs) noexcept {
		std::istream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Base64_istream &lhs, Base64_istream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

class Base64_ostream : public std::ostream {
private:
	Base64_streambuf m_sb;

public:
	explicit Base64_ostream(std::ios_base::openmode which = std::ios_base::out)
		: std::ostream(&m_sb)
		, m_sb(which | std::ios_base::out)
	{
	}
	explicit Base64_ostream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::out)
		: std::ostream(&m_sb)
		, m_sb(STD_MOVE(buffer), which | std::ios_base::out)
	{
	}
	~Base64_ostream() OVERRIDE;

public:
	Base64_streambuf *rdbuf() const {
		return const_cast<Base64_streambuf *>(&m_sb);
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
	void swap(Base64_ostream &rhs) noexcept {
		std::ostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Base64_ostream &lhs, Base64_ostream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

class Base64_stream : public std::iostream {
private:
	Base64_streambuf m_sb;

public:
	explicit Base64_stream(std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::iostream(&m_sb)
		, m_sb(which)
	{
	}
	explicit Base64_stream(StreamBuffer buffer, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::iostream(&m_sb)
		, m_sb(STD_MOVE(buffer), which)
	{
	}
	~Base64_stream() OVERRIDE;

public:
	Base64_streambuf *rdbuf() const {
		return const_cast<Base64_streambuf *>(&m_sb);
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
	void swap(Base64_stream &rhs) noexcept {
		std::iostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Base64_stream &lhs, Base64_stream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

}

#endif
