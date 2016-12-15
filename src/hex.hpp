// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HEX_HPP_
#define POSEIDON_HEX_HPP_

#include "cxx_ver.hpp"
#include <streambuf>
#include <istream>
#include <ostream>
#include "stream_buffer.hpp"

namespace Poseidon {

class Hex_streambuf : public std::streambuf {
private:
	StreamBuffer m_buffer;
	bool m_upper_case;
	std::ios_base::openmode m_which;
	char m_get_area[1];

public:
	explicit Hex_streambuf(std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::streambuf()
		, m_buffer(), m_upper_case(false), m_which(which)
	{
	}
	explicit Hex_streambuf(StreamBuffer buffer, bool upper_case = false, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::streambuf()
		, m_buffer(STD_MOVE(buffer)), m_upper_case(upper_case), m_which(which)
	{
	}
	Hex_streambuf(const Hex_streambuf &rhs)
		: std::streambuf()
		, m_buffer(rhs.m_buffer), m_upper_case(rhs.m_upper_case), m_which(rhs.m_which)
	{
	}
	Hex_streambuf &operator=(const Hex_streambuf &rhs){
		sync();
		m_buffer = rhs.m_buffer;
		m_upper_case = rhs.m_upper_case;
		m_which = rhs.m_which;
		return *this;
	}
#ifdef POSEIDON_CXX11
	Hex_streambuf(Hex_streambuf &&rhs) noexcept
		: std::streambuf()
		, m_buffer(std::move(rhs.m_buffer)), m_upper_case(rhs.m_upper_case), m_which(rhs.m_which)
	{
	}
	Hex_streambuf &operator=(Hex_streambuf &&rhs) noexcept {
		sync();
		m_buffer = std::move(rhs.m_buffer);
		m_upper_case = rhs.m_upper_case;
		m_which = rhs.m_which;
		return *this;
	}
#endif
	~Hex_streambuf() OVERRIDE;

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

	bool is_upper_case() const {
		return m_upper_case;
	}
	void set_upper_case(bool upper_case){
		m_upper_case = upper_case;
	}

	void swap(Hex_streambuf &rhs) NOEXCEPT {
		sync();
		using std::swap;
		swap(m_buffer, rhs.m_buffer);
		swap(m_upper_case, rhs.m_upper_case);
		swap(m_which, rhs.m_which);
	}
};

inline void swap(Hex_streambuf &lhs, Hex_streambuf &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

class Hex_istream : public std::istream {
private:
	Hex_streambuf m_sb;

public:
	explicit Hex_istream(std::ios_base::openmode which = std::ios_base::in)
		: std::istream(&m_sb)
		, m_sb(which | std::ios_base::in)
	{
	}
	explicit Hex_istream(StreamBuffer buffer, bool upper_case = false, std::ios_base::openmode which = std::ios_base::in)
		: std::istream(&m_sb)
		, m_sb(STD_MOVE(buffer), upper_case, which | std::ios_base::in)
	{
	}
	~Hex_istream() OVERRIDE;

public:
	Hex_streambuf *rdbuf() const {
		return const_cast<Hex_streambuf *>(&m_sb);
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

	bool is_upper_case() const {
		return rdbuf()->is_upper_case();
	}
	void set_upper_case(bool upper_case){
		rdbuf()->set_upper_case(upper_case);
	}

#ifdef POSEIDON_CXX14
	void swap(Hex_istream &rhs) noexcept {
		std::istream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Hex_istream &lhs, Hex_istream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

class Hex_ostream : public std::ostream {
private:
	Hex_streambuf m_sb;

public:
	explicit Hex_ostream(std::ios_base::openmode which = std::ios_base::out)
		: std::ostream(&m_sb)
		, m_sb(which | std::ios_base::out)
	{
	}
	explicit Hex_ostream(StreamBuffer buffer, bool upper_case = false, std::ios_base::openmode which = std::ios_base::out)
		: std::ostream(&m_sb)
		, m_sb(STD_MOVE(buffer), upper_case, which | std::ios_base::out)
	{
	}
	~Hex_ostream() OVERRIDE;

public:
	Hex_streambuf *rdbuf() const {
		return const_cast<Hex_streambuf *>(&m_sb);
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

	bool is_upper_case() const {
		return rdbuf()->is_upper_case();
	}
	void set_upper_case(bool upper_case){
		rdbuf()->set_upper_case(upper_case);
	}

#ifdef POSEIDON_CXX14
	void swap(Hex_ostream &rhs) noexcept {
		std::ostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Hex_ostream &lhs, Hex_ostream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

class Hex_stream : public std::iostream {
private:
	Hex_streambuf m_sb;

public:
	explicit Hex_stream(std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::iostream(&m_sb)
		, m_sb(which)
	{
	}
	explicit Hex_stream(StreamBuffer buffer, bool upper_case = false, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
		: std::iostream(&m_sb)
		, m_sb(STD_MOVE(buffer), upper_case, which)
	{
	}
	~Hex_stream() OVERRIDE;

public:
	Hex_streambuf *rdbuf() const {
		return const_cast<Hex_streambuf *>(&m_sb);
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

	bool is_upper_case() const {
		return rdbuf()->is_upper_case();
	}
	void set_upper_case(bool upper_case){
		rdbuf()->set_upper_case(upper_case);
	}

#ifdef POSEIDON_CXX14
	void swap(Hex_stream &rhs) noexcept {
		std::iostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Hex_stream &lhs, Hex_stream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

}

#endif
