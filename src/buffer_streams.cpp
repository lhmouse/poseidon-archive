// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "buffer_streams.hpp"

namespace Poseidon {

Buffer_streambuf::~Buffer_streambuf(){ }

int Buffer_streambuf::sync(){
	if(gptr()){
		m_buffer.discard(static_cast<std::size_t>(gptr() - eback()));
		setg(NULLPTR, NULLPTR, NULLPTR);
	}
	return std::streambuf::sync();
}

std::streamsize Buffer_streambuf::showmanyc(){
	if(m_which & std::ios_base::in){
		std::streamsize n_avail = static_cast<std::streamsize>(m_buffer.size());
		if(gptr()){
			n_avail -= gptr() - eback();
		}
		return n_avail;
	} else {
		return 0;
	}
}
std::streamsize Buffer_streambuf::xsgetn(Buffer_streambuf::char_type *s, std::streamsize n){
	if(m_which & std::ios_base::in){
		sync();
		std::streamsize n_got = std::min<std::streamsize>(n, PTRDIFF_MAX);
		n_got = static_cast<std::streamsize>(m_buffer.get(s, static_cast<std::size_t>(n_got)));
		return n_got;
	} else {
		return 0;
	}
}
Buffer_streambuf::int_type Buffer_streambuf::underflow(){
	if(m_which & std::ios_base::in){
		sync();
		std::streamsize n_peeked = static_cast<std::streamsize>(m_buffer.peek(m_get_area.begin(), m_get_area.size()));
		if(n_peeked == 0){
			return traits_type::eof();
		}
		setg(m_get_area.begin(), m_get_area.begin(), m_get_area.begin() + n_peeked);
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
		sync();
		m_buffer.unget(traits_type::to_char_type(c));
		return c;
	} else {
		return traits_type::eof();
	}
}

std::streamsize Buffer_streambuf::xsputn(const Buffer_streambuf::char_type *s, std::streamsize n){
	if(m_which & std::ios_base::out){
		std::streamsize n_put = std::min<std::streamsize>(n, PTRDIFF_MAX);
		m_buffer.put(s, static_cast<std::size_t>(n_put));
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
		m_buffer.put(traits_type::to_char_type(c));
		return c;
	} else {
		return traits_type::eof();
	}
}

Buffer_istream::~Buffer_istream(){ }

Buffer_ostream::~Buffer_ostream(){ }

Buffer_stream::~Buffer_stream(){ }

}
