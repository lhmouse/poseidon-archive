// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "multipart.hpp"
#include "../profiler.hpp"
#include "../log.hpp"
#include "../protocol_exception.hpp"
#include "../random.hpp"
#include "../string.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {
namespace Http {

namespace {
	CONSTEXPR const char     BOUNDARY_CHARS[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
	CONSTEXPR const unsigned BOUNDARY_LEN     = 60;

	const Http::MultipartElement g_empty_multipart_element;

	inline bool try_pop(std::string &str, char ch){
		if(str.empty()){
			return false;
		}
#ifdef POSEIDON_CXX11
		if(str.back() != ch){
			return false;
		}
		str.pop_back();
#else
		if(*str.rbegin() != ch){
			return false;
		}
		str.erase(str.end() - 1);
#endif
		return true;
	}
}

const MultipartElement &empty_multipart_element() NOEXCEPT {
	return g_empty_multipart_element;
}

Multipart::Multipart(std::string boundary, std::istream &is)
	: m_boundary(STD_MOVE(boundary)), m_elements()
{
	parse(is);
	DEBUG_THROW_UNLESS(is, ProtocolException, sslit("Http::Multipart parser error"), -1);
}

void Multipart::random_boundary(){
	PROFILE_ME;

	std::string boundary;
	boundary.reserve(BOUNDARY_LEN);
	for(unsigned i = 0; i < BOUNDARY_LEN; ++i){
		boundary.push_back(BOUNDARY_CHARS[random_uint32() % (sizeof(BOUNDARY_CHARS) - 1)]);
	}
	m_boundary.swap(boundary);
}

std::string Multipart::dump() const {
	PROFILE_ME;

	Buffer_ostream os;
	dump(os);
	return os.get_buffer().dump_string();
}
void Multipart::dump(std::ostream &os) const {
	PROFILE_ME;
	DEBUG_THROW_UNLESS(!m_boundary.empty(), ProtocolException, sslit("Multipart boundary not set"), -1);

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
	PROFILE_ME;
	DEBUG_THROW_UNLESS(!m_boundary.empty(), ProtocolException, sslit("Multipart boundary not set"), -1);

	const AUTO(window_size, m_boundary.size() + 2);

	VALUE_TYPE(m_elements) elements;

	StreamBuffer buffer;
	std::string queue;
	enum {
		S_INIT,
		S_BOUNDARY_END,
		S_BOUNDARY_END_WANTS_LF,
		S_SEGMENT,
		S_FINISH_WANTS_HYPHEN,
		S_FINISH,
	} state = S_INIT;
	typedef std::istream::traits_type traits;
	traits::int_type next = is.peek();
	for(; !traits::eq_int_type(next, traits::eof()); next = is.peek()){
		const char ch = traits::to_char_type(is.get());
		switch(state){
		case S_INIT:
			queue.push_back(ch);
			if(queue.size() < window_size){
				break;
			}
			queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(queue.size() - window_size));
			if((queue[0] != '-') || (queue[1] != '-') || (queue.compare(2, queue.npos, m_boundary) != 0)){
				break;
			}
			queue.clear();
			state = S_BOUNDARY_END;
			break;

		case S_BOUNDARY_END:
			if(ch == '\r'){
				state = S_BOUNDARY_END_WANTS_LF;
				break;
			}
			if(ch == '-'){
				state = S_FINISH_WANTS_HYPHEN;
				break;
			}
			if(ch == '\n'){
				state = S_SEGMENT;
				break;
			}
			LOG_POSEIDON_WARNING("Invalid multipart boundary.");
			is.setstate(std::ios::failbit);
			return;

		case S_BOUNDARY_END_WANTS_LF:
			if(ch == '\n'){
				state = S_SEGMENT;
				break;
			}
			LOG_POSEIDON_WARNING("Invalid multipart boundary.");
			is.setstate(std::ios::failbit);
			return;

		case S_SEGMENT: {
			queue.push_back(ch);
			if(queue.size() < window_size){
				break;
			}
			const AUTO(offset, queue.size() - window_size);
			if((queue[offset] != '-') || (queue[offset + 1] != '-') || (queue.compare(offset + 2, queue.npos, m_boundary) != 0)){
				break;
			}
			queue.erase(queue.end() - static_cast<std::ptrdiff_t>(window_size), queue.end());

			MultipartElement elem;
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
				DEBUG_THROW_UNLESS(pos != std::string::npos, ProtocolException, sslit("Invalid HTTP header"), -1);
				SharedNts key(line.data(), pos);
				line.erase(0, pos + 1);
				std::string value(trim(STD_MOVE(line)));
				elem.headers.set(STD_MOVE(key), STD_MOVE(value));
			}
			if(try_pop(queue, '\n')){
				try_pop(queue, '\r');
			}
			elem.entity = StreamBuffer(queue);
			elements.push_back(STD_MOVE(elem));

			queue.clear();
			state = S_BOUNDARY_END;
			break; }

		case S_FINISH_WANTS_HYPHEN:
			if(ch == '-'){
				state = S_FINISH;
				break;
			}
			LOG_POSEIDON_WARNING("Invalid multipart termination boundary.");
			is.setstate(std::ios::failbit);
			return;

		case S_FINISH:
			break;
		}
		if(state == S_FINISH){
			next = is.peek();
			break;
		}
	}

	m_elements.swap(elements);
}

}
}
