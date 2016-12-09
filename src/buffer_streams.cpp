// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "buffer_streams.hpp"

namespace Poseidon {

Buffer_streambuf::~Buffer_streambuf(){
}

void Buffer_streambuf::imbue(const std::locale &locale){
	std::streambuf::imbue(locale);
}

Buffer_streambuf *Buffer_streambuf::setbuf(Buffer_streambuf::char_type *s, std::streamsize n){
	std::streambuf::setbuf(s, n);
	return this;
}
Buffer_streambuf::pos_type Buffer_streambuf::seekoff(Buffer_streambuf::off_type off, std::ios_base::seekdir way, std::ios_base::openmode which){
	return std::streambuf::seekoff(off, way, which);
}
Buffer_streambuf::pos_type Buffer_streambuf::seekpos(Buffer_streambuf::pos_type sp, std::ios_base::openmode which){
	return std::streambuf::seekpos(sp, which);
}
int Buffer_streambuf::sync(){
	return std::streambuf::sync();
}

std::streamsize Buffer_streambuf::showmanyc(){
	if(!(m_which & std::ios_base::in)){
		return 0;
	}
	return static_cast<std::streamsize>(m_buffer.size());
}
std::streamsize Buffer_streambuf::xsgetn(Buffer_streambuf::char_type *s, std::streamsize n){
	if(!(m_which & std::ios_base::in)){
		return 0;
	}
	unsigned n_got = static_cast<unsigned>(std::min<std::streamsize>(n, INT_MAX));
	n_got = m_buffer.get(s, n_got);
	return static_cast<std::streamsize>(n_got);
}
Buffer_streambuf::int_type Buffer_streambuf::underflow(){
	if(!(m_which & std::ios_base::in)){
		return traits_type::eof();
	}
	return m_buffer.empty() ? traits_type::eof() : m_buffer.peek();
}
Buffer_streambuf::int_type Buffer_streambuf::uflow(){
	if(!(m_which & std::ios_base::in)){
		return traits_type::eof();
	}
	return m_buffer.empty() ? traits_type::eof() : m_buffer.get();
}

Buffer_streambuf::int_type Buffer_streambuf::pbackfail(Buffer_streambuf::int_type c){
	if(!(m_which & std::ios_base::out)){
		return traits_type::eof();
	}
	return traits_type::eq_int_type(c, traits_type::eof()) ? traits_type::eof() : (m_buffer.unget(c), c);
}

std::streamsize Buffer_streambuf::xsputn(const Buffer_streambuf::char_type *s, std::streamsize n){
	if(!(m_which & std::ios_base::out)){
		return 0;
	}
	unsigned n_put = static_cast<unsigned>(std::min<std::streamsize>(n, INT_MAX));
	m_buffer.put(s, n_put);
	return static_cast<std::streamsize>(n_put);
}
Buffer_streambuf::int_type Buffer_streambuf::overflow(Buffer_streambuf::int_type c){
	if(!(m_which & std::ios_base::out)){
		return traits_type::eof();
	}
	return traits_type::eq_int_type(c, traits_type::eof()) ? traits_type::not_eof(c) : (m_buffer.put(c), c);
}

Buffer_istream::~Buffer_istream(){
}
Buffer_ostream::~Buffer_ostream(){
}
Buffer_iostream::~Buffer_iostream(){
}

}
