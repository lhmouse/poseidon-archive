// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "buffer_streams.hpp"

namespace Poseidon {

Buffer_streambuf::~Buffer_streambuf(){
}

std::size_t Buffer_streambuf::cast_size(std::streamsize n){
	if(n <= 0){
		return 0;
	} else if(n >= UINT_MAX){
		return UINT_MAX;
	} else {
		return static_cast<std::size_t>(n);
	}
}

bool Buffer_streambuf::sync_in(){
	if(gptr()){
		for(AUTO(p, egptr()); p != gptr(); --p){
			m_buffer.unget(static_cast<unsigned char>(p[-1]));
		}
		setg(m_get_area.begin(), m_get_area.end(), m_get_area.end());
		return true;
	} else {
		return false;
	}
}
bool Buffer_streambuf::sync_out(){
	if(pptr()){
		m_buffer.put(pptr(), static_cast<std::size_t>(pptr() - pbase()));
		setp(m_put_area.begin(), m_put_area.end());
		return true;
	} else {
		return false;
	}
}

void Buffer_streambuf::imbue(const std::locale &locale){
	sync();
	std::streambuf::imbue(locale);
}

Buffer_streambuf *Buffer_streambuf::setbuf(std::streambuf::char_type *s, std::streamsize n){
	sync();
	std::streambuf::setbuf(s, n);
	return this;
}
std::streambuf::pos_type Buffer_streambuf::seekoff(std::streambuf::off_type off, std::ios_base::seekdir way, std::ios_base::openmode which){
	sync();
	return std::streambuf::seekoff(off, way, which);
}
std::streambuf::pos_type Buffer_streambuf::seekpos(std::streambuf::pos_type sp, std::ios_base::openmode which){
	sync();
	return std::streambuf::seekpos(sp, which);
}
int Buffer_streambuf::sync(){
	if(m_which & std::ios_base::in){
		sync_in();
	}
	if(m_which & std::ios_base::out){
		sync_out();
	}
	return std::streambuf::sync();
}

std::streamsize Buffer_streambuf::showmanyc(){
	if(m_which & std::ios_base::in){
		std::streamsize n = 0;
		if(gptr()){
			n += egptr() - gptr();
		}
		n += static_cast<std::streamsize>(m_buffer.size());
		return n;
	} else {
		return 0;
	}
}
std::streamsize Buffer_streambuf::xsgetn(std::streambuf::char_type *s, std::streamsize n){
	if(m_which & std::ios_base::in){
		sync_in();
		std::size_t size = cast_size(n);
		size = m_buffer.get(s, size);
		return static_cast<std::streamsize>(size);
	} else {
		return 0;
	}
}
std::streambuf::int_type Buffer_streambuf::underflow(){
	sync_out();
	VALUE_TYPE(m_get_area) temp;
	const unsigned bytes_read = m_buffer.get(temp.data(), temp.size());
	if(bytes_read != 0){
		std::memcpy(m_get_area.end() - bytes_read, temp.data(), bytes_read);
		setg(m_get_area.begin(), m_get_area.end() - bytes_read, m_get_area.end());
		return traits_type::to_int_type(*gptr());
	} else {
		return traits_type::eof();
	}
}
std::streambuf::int_type Buffer_streambuf::uflow(){
	return std::streambuf::uflow();
}

std::streambuf::int_type Buffer_streambuf::pbackfail(std::streambuf::int_type c){
	if(m_which & std::ios_base::out){
		if(traits_type::eq_int_type(c, traits_type::eof())){
			return traits_type::eof();
		} else {
			sync_in();
			m_buffer.unget(c);
			return c;
		}
	} else {
		return traits_type::eof();
	}
}

std::streamsize Buffer_streambuf::xsputn(const std::streambuf::char_type *s, std::streamsize n){
	if(m_which & std::ios_base::out){
		sync_out();
		std::size_t size = cast_size(n);
		m_buffer.put(s, size);
		return static_cast<std::streamsize>(size);
	} else {
		return 0;
	}
}
std::streambuf::int_type Buffer_streambuf::overflow(std::streambuf::int_type c){
	if(m_which & std::ios_base::out){
		if(traits_type::eq_int_type(c, traits_type::eof())){
			sync_out();
			return traits_type::not_eof(c);
		} else {
			sync_out();
			m_buffer.put(c);
			return c;
		}
	} else {
		return traits_type::eof();
	}
}

Buffer_istream::~Buffer_istream(){
}
Buffer_ostream::~Buffer_ostream(){
}
Buffer_iostream::~Buffer_iostream(){
}

}
