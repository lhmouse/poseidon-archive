// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "multipart.hpp"
#include "utilities.hpp"
#include "../profiler.hpp"
#include "../log.hpp"
#include "../protocol_exception.hpp"
#include "../random.hpp"
#include "../string.hpp"
#include <string.h>

namespace Poseidon {

namespace {
	const Http::MultipartElement g_empty_multipart_element;
}

namespace Http {
	const MultipartElement &empty_multipart_element() NOEXCEPT {
		return g_empty_multipart_element;
	}

	void Multipart::random_boundary(){
		PROFILE_ME;

		static const char s_boundary_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
		static const std::size_t s_boundary_len = 60;

		std::string boundary;
		boundary.reserve(s_boundary_len);
		for(std::size_t i = 0; i < s_boundary_len; ++i){
			boundary.push_back(s_boundary_chars[Poseidon::random_uint32() % (sizeof(s_boundary_chars) - 1)]);
		}
		m_boundary.swap(boundary);
	}

	std::string Multipart::dump() const {
		PROFILE_ME;

		std::ostringstream oss;
		dump(oss);
		return oss.str();
	}
	void Multipart::dump(std::ostream &os) const {
		PROFILE_ME;

		if(m_boundary.empty()){
			DEBUG_THROW(ProtocolException, sslit("Multipart boundary not set"), -1);
		}

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

		if(m_boundary.empty()){
			DEBUG_THROW(ProtocolException, sslit("Multipart boundary not set"), -1);
		}

		VALUE_TYPE(m_elements) elements;

		typedef std::istream::traits_type traits;
		MultipartElement elem;
		std::string line;
		enum {
			S_INIT,
			S_INIT_LEADING_HYPHEN_ONE,
			S_INIT_BOUNDARY,
			S_BOUNDARY_END,
			S_HEADERS,
			S_ENTITY,
			S_LEADING_HYPHEN_ONE,
			S_BOUNDARY,
			S_TRAILING_HYPHEN_ONE,
		} state = S_INIT;
		char ch;
		while(!traits::eq_int_type(is.peek(), traits::eof()) && is.get(ch)){
			switch(state){
			case S_INIT:
				if(ch == '-'){
					state = S_INIT_LEADING_HYPHEN_ONE;
					break;
				}
				// state = S_INIT;
				break;

			case S_INIT_LEADING_HYPHEN_ONE:
				if(ch == '-'){
					line.clear();
					state = S_INIT_BOUNDARY;
					break;
				}
				state = S_INIT;
				break;

			case S_INIT_BOUNDARY:
				 line.push_back(ch);
				 if(std::memcmp(line.data(), m_boundary.data(), line.size()) == 0){
					if(line.size() != m_boundary.size()){
						// state = S_INIT_BOUNDARY;
						break;
					}
					state = S_BOUNDARY_END;
					break;
				}
				state = S_INIT;
				break;

			case S_BOUNDARY_END:
				if(ch == '-'){
					state = S_TRAILING_HYPHEN_ONE;
					break;
				}
				if(ch == '\n'){
					if(!line.empty() && (*line.rbegin() == '\r')){
						line.erase(line.end() - 1);
					}
					if(line != m_boundary){
						LOG_POSEIDON_WARNING("Invalid multipart boundary.");
						is.setstate(std::ios::failbit);
						return;
					}
					elem = VAL_INIT;
					line.clear();
					state = S_HEADERS;
					break;
				}
				LOG_POSEIDON_WARNING("Invalid multipart boundary.");
				is.setstate(std::ios::failbit);
				return;

			case S_HEADERS:
				if(ch == '\n'){
					if(!line.empty() && (*line.rbegin() == '\r')){
						line.erase(line.end() - 1);
					}
					if(!line.empty()){
						AUTO(pos, line.find(':'));
						if(pos == std::string::npos){
							LOG_POSEIDON_WARNING("Invalid HTTP header: ", line);
							is.setstate(std::ios::failbit);
							return;
						}
						SharedNts key(line.data(), pos);
						line.erase(0, pos + 1);
						std::string value(ltrim(STD_MOVE(line)));
						elem.headers.append(STD_MOVE(key), STD_MOVE(value));
						line.clear();
						// state = S_HEADERS;
						break;
					} else {
						state = S_ENTITY;
						break;
					}
				}
				line.push_back(ch);
				// state = S_HEADERS;
				break;

			case S_ENTITY:
				if(ch == '-'){
					state = S_LEADING_HYPHEN_ONE;
					break;
				}
				elem.entity.put((unsigned char)ch);
				// state = S_ENTITY;
				break;

			case S_LEADING_HYPHEN_ONE:
				if(ch == '-'){
					line.clear();
					state = S_BOUNDARY;
					break;
				}
				elem.entity.put('-');
				elem.entity.put((unsigned char)ch);
				state = S_ENTITY;
				break;

			case S_BOUNDARY:
				 line.push_back(ch);
				 if(std::memcmp(line.data(), m_boundary.data(), line.size()) == 0){
					if(line.size() != m_boundary.size()){
						// state = S_BOUNDARY;
						break;
					}
					if(elem.entity.back() == '\n'){
						elem.entity.unput();
						if(elem.entity.back() == '\r'){
							elem.entity.unput();
						}
					}
					elements.push_back(STD_MOVE(elem));
					state = S_BOUNDARY_END;
					break;
				}
				elem.entity.put('-');
				elem.entity.put(line);
				state = S_ENTITY;
				break;

			case S_TRAILING_HYPHEN_ONE:
				if(ch == '-'){
					goto _done;
				}
				LOG_POSEIDON_WARNING("Invalid multipart termination boundary.");
				is.setstate(std::ios::failbit);
				return;
			}
		}
	_done:
		;

		m_elements.swap(elements);

		if(m_elements.empty()){
			is.setstate(std::ios::failbit);
		}
	}
}

}
