// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "buffer_streams.hpp"

namespace Poseidon {

Buffer_streambuf::~Buffer_streambuf(){
}

int Buffer_streambuf::sync(){
	setg(m_get_area.begin(), m_get_area.end(), m_get_area.end());
	return std::streambuf::sync();
}

std::streamsize Buffer_streambuf::showmanyc(){
	if(m_which & std::ios_base::in){
		return static_cast<std::streamsize>(m_buffer.size());
	} else {
		return 0;
	}
}
std::streamsize Buffer_streambuf::xsgetn(Buffer_streambuf::char_type *s, std::streamsize n){
	if(m_which & std::ios_base::in){
		int n_got = std::min<std::streamsize>(n, INT_MAX);
		n_got = static_cast<int>(m_buffer.get(s, static_cast<unsigned>(n_got)));
		if(gptr() && (egptr() - gptr() > n_got)){
			setg(m_get_area.begin(), gptr() + n_got, m_get_area.end());
		} else {
			setg(m_get_area.begin(), m_get_area.end(), m_get_area.end());
		}
		return n_got;
	} else {
		return 0;
	}
}
Buffer_streambuf::int_type Buffer_streambuf::underflow(){
	if(m_which & std::ios_base::in){
		char temp[sizeof(m_get_area)];
		int n_peeked = static_cast<int>(m_buffer.peek(temp, sizeof(temp)));
		if(n_peeked == 0){
			return traits_type::eof();
		}
		setg(m_get_area.begin(), m_get_area.end() - n_peeked, m_get_area.end());
		std::copy(temp, temp + n_peeked, gptr());
		return traits_type::to_int_type(*gptr());
	} else {
		return traits_type::eof();
	}
}

Buffer_streambuf::int_type Buffer_streambuf::pbackfail(Buffer_streambuf::int_type c){
	if(m_which & std::ios_base::out){
		if(traits_type::eq_int_type(c, traits_type::eof())){
			return traits_type::eof();
		}
		m_buffer.unget(c);
		if(gptr()){
			std::copy_backward(gptr(), egptr() - 1, egptr());
			*gptr() = c;
		}
		return c;
	} else {
		return traits_type::eof();
	}
}

std::streamsize Buffer_streambuf::xsputn(const Buffer_streambuf::char_type *s, std::streamsize n){
	if(m_which & std::ios_base::out){
		int n_put = std::min<std::streamsize>(n, INT_MAX);
		m_buffer.put(s, static_cast<unsigned>(n_put));
		return n_put;
	} else {
		return 0;
	}
}
Buffer_streambuf::int_type Buffer_streambuf::overflow(Buffer_streambuf::int_type c){
	if(m_which & std::ios_base::out){
		if(traits_type::eq_int_type(c, traits_type::eof())){
			return traits_type::not_eof(c);
		}
		m_buffer.put(c);
		return c;
	} else {
		return traits_type::eof();
	}
}

Buffer_istream::~Buffer_istream(){
}
Buffer_ostream::~Buffer_ostream(){
}
Buffer_stream::~Buffer_stream(){
}

}
