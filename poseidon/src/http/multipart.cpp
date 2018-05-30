// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "multipart.hpp"
#include "../profiler.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../random.hpp"
#include "../string.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {
namespace Http {

namespace {
	CONSTEXPR const char     s_boundary_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
	CONSTEXPR const unsigned s_boundary_len     = 60;

	const Http::Multipart_element g_empty_multipart_element;

	inline bool try_pop(std::string &str, char ch){
		if(str.empty()){
			return false;
		}
		if(*str.rbegin() != ch){
			return false;
		}
		str.erase(str.end() - 1);
		return true;
	}
}

const Multipart_element & empty_multipart_element() NOEXCEPT {
	return g_empty_multipart_element;
}

Multipart::Multipart(std::string boundary, std::istream &is)
	: m_boundary(STD_MOVE(boundary)), m_elements()
{
	parse(is);
	POSEIDON_THROW_UNLESS(is, Basic_exception, Rcnts::view("Http::Multipart parser error"));
}

void Multipart::random_boundary(){
	POSEIDON_PROFILE_ME;

	std::string boundary;
	boundary.reserve(s_boundary_len);
	for(unsigned i = 0; i < s_boundary_len; ++i){
		boundary.push_back(s_boundary_chars[random_uint32() % (sizeof(s_boundary_chars) - 1)]);
	}
	m_boundary.swap(boundary);
}

Stream_buffer Multipart::dump() const {
	POSEIDON_PROFILE_ME;

	Buffer_ostream bos;
	dump(bos);
	return STD_MOVE(bos.get_buffer());
}
void Multipart::dump(std::ostream &os) const {
	POSEIDON_PROFILE_ME;
	POSEIDON_THROW_UNLESS(!m_boundary.empty(), Basic_exception, Rcnts::view("Multipart boundary not set"));

	os <<"--" <<m_boundary;
	for(AUTO(it, m_elements.begin()); it != m_elements.end(); ++it){
		for(AUTO(zit, it->headers.begin()); zit != it->headers.end(); ++zit){
			os <<zit->first <<": " <<zit->second <<"\r\n";
		}
		os <<"\r\n";
		os <<it->entity;
		os <<"\r\n--" <<m_boundary;
	}
	os <<"--\r\n";
}
void Multipart::parse(std::istream &is){
	POSEIDON_PROFILE_ME;
	POSEIDON_THROW_UNLESS(!m_boundary.empty(), Basic_exception, Rcnts::view("Multipart boundary not set"));

	const AUTO(window_size, m_boundary.size() + 2);

	VALUE_TYPE(m_elements) elements;

	Stream_buffer buffer;
	std::string queue;
	enum {
		state_init,
		state_boundary_end,
		state_boundary_end_wants_lf,
		state_segment,
		state_finish_wants_hyphen,
		state_finish,
	} state = state_init;
	typedef std::istream::traits_type traits;
	traits::int_type next = is.peek();
	for(; !traits::eq_int_type(next, traits::eof()); next = is.peek()){
		const char ch = traits::to_char_type(is.get());
		switch(state){
		case state_init:
			queue.push_back(ch);
			if(queue.size() < window_size){
				break;
			}
			queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(queue.size() - window_size));
			if((queue[0] != '-') || (queue[1] != '-') || (queue.compare(2, queue.npos, m_boundary) != 0)){
				break;
			}
			queue.clear();
			state = state_boundary_end;
			break;

		case state_boundary_end:
			if(ch == '\r'){
				state = state_boundary_end_wants_lf;
				break;
			}
			if(ch == '-'){
				state = state_finish_wants_hyphen;
				break;
			}
			if(ch == '\n'){
				state = state_segment;
				break;
			}
			POSEIDON_LOG_WARNING("Invalid multipart boundary.");
			is.setstate(std::ios::failbit);
			return;

		case state_boundary_end_wants_lf:
			if(ch == '\n'){
				state = state_segment;
				break;
			}
			POSEIDON_LOG_WARNING("Invalid multipart boundary.");
			is.setstate(std::ios::failbit);
			return;

		case state_segment: {
			queue.push_back(ch);
			if(queue.size() < window_size){
				break;
			}
			const AUTO(offset, queue.size() - window_size);
			if((queue[offset] != '-') || (queue[offset + 1] != '-') || (queue.compare(offset + 2, queue.npos, m_boundary) != 0)){
				break;
			}
			queue.erase(queue.end() - static_cast<std::ptrdiff_t>(window_size), queue.end());

			Multipart_element elem;
			std::string line;
			for(;;){
				AUTO(pos, queue.find('\n'));
				if(pos == std::string::npos){
					line = STD_MOVE(queue);
					queue.clear();
				} else {
					line = queue.substr(0, pos);
					queue.erase(0, pos + 1);
				}
				try_pop(line, '\r');
				if(line.empty()){
					break;
				}
				pos = line.find(':');
				POSEIDON_THROW_UNLESS(pos != std::string::npos, Basic_exception, Rcnts::view("Invalid HTTP header"));
				Rcnts key(line.data(), pos);
				line.erase(0, pos + 1);
				std::string value(trim(STD_MOVE(line)));
				elem.headers.set(STD_MOVE(key), STD_MOVE(value));
			}
			if(try_pop(queue, '\n')){
				try_pop(queue, '\r');
			}
			elem.entity = Stream_buffer(queue);
			elements.push_back(STD_MOVE(elem));

			queue.clear();
			state = state_boundary_end;
			break; }

		case state_finish_wants_hyphen:
			if(ch == '-'){
				state = state_finish;
				break;
			}
			POSEIDON_LOG_WARNING("Invalid multipart termination boundary.");
			is.setstate(std::ios::failbit);
			return;

		case state_finish:
			break;
		}
		if(state == state_finish){
			next = is.peek();
			break;
		}
	}

	m_elements.swap(elements);
}

}
}
