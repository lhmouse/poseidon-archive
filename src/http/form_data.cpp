// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "form_data.hpp"
#include "utilities.hpp"
#include "../profiler.hpp"
#include "../log.hpp"
#include "../protocol_exception.hpp"
#include "../random.hpp"
#include "../string.hpp"
#include <string.h>

namespace Poseidon {

namespace {
	const Http::FormDataElement g_empty_form_data_element;
}

namespace Http {
	const FormDataElement &empty_form_data_element() NOEXCEPT {
		return g_empty_form_data_element;
	}

	void FormData::random_boundary(){
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

	std::string FormData::dump() const {
		PROFILE_ME;

		std::ostringstream oss;
		dump(oss);
		return oss.str();
	}
	void FormData::dump(std::ostream &os) const {
		PROFILE_ME;

		if(m_boundary.empty()){
			DEBUG_THROW(ProtocolException, sslit("Multipart boundary not set"), -1);
		}

		os <<"--" <<m_boundary;
		for(AUTO(it, m_elements.begin()); it != m_elements.end(); ++it){
			const AUTO_REF(name, it->first);
			const AUTO_REF(elem, it->second);
			HeaderOption content_disposition;
			content_disposition.set_base("form-data");
			content_disposition.set_option(sslit("name"), url_encode(name));
			if(!elem.filename.empty()){
				content_disposition.set_option(sslit("filename"), url_encode(elem.filename));
			}
			os <<"\r\nContent-Disposition: " <<content_disposition <<"\r\n";
			for(AUTO(zit, elem.headers.begin()); zit != elem.headers.end(); ++zit){
				if(zit->first == "Content-Disposition"){
					continue;
				}
				os <<zit->first <<": " <<zit->second <<"\r\n";
			}
			os <<"\r\n";
			os <<elem.entity;
			os <<"\r\n--" <<m_boundary;
		}
		os <<"--\r\n";
	}
	void FormData::parse(std::istream &is){
		PROFILE_ME;

		if(m_boundary.empty()){
			DEBUG_THROW(ProtocolException, sslit("Multipart boundary not set"), -1);
		}

		VALUE_TYPE(m_elements) elements;

		SharedNts name;
		FormDataElement elem;

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
		while(is.get(ch)){
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
						is.setstate(std::istream::failbit);
						return;
					}
					name = VAL_INIT;
					elem = VAL_INIT;
					line.clear();
					state = S_HEADERS;
					break;
				}
				LOG_POSEIDON_WARNING("Invalid multipart boundary.");
				is.setstate(std::istream::failbit);
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
							is.setstate(std::istream::failbit);
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
						const AUTO_REF(str, elem.headers.get("Content-Disposition"));
						std::istringstream iss(str);
						Http::HeaderOption content_disposition(iss);
						if(!iss){
							LOG_POSEIDON_WARNING("Invalid Content-Disposition: ", str);
							is.setstate(std::istream::failbit);
							return;
						}
						if(::strcasecmp(content_disposition.get_base().c_str(), "form-data") == 0){
							name = SharedNts(content_disposition.get_option("name"));
							elem.filename = STD_MOVE(content_disposition.get_option("filename"));
						} else {
							LOG_POSEIDON_WARNING("Non-form-data ignored: content_disposition = ", content_disposition);
						}
						elem.headers.erase("Content-Disposition");
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
					if(!name.empty()){
						elements[STD_MOVE(name)] = STD_MOVE(elem);
					}
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
				is.setstate(std::istream::failbit);
				return;
			}
		}
	_done:
		;

		m_elements.swap(elements);
	}
}

}
