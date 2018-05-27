// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "json.hpp"
#include "log.hpp"
#include "profiler.hpp"
#include "buffer_streams.hpp"
#include "exception.hpp"

namespace Poseidon {

namespace {
	const Json_element g_null_element = Json_null();

	extern Json_element accept_element(std::istream &is);

	std::string accept_string(std::istream &is){
		POSEIDON_PROFILE_ME;

		std::string ret;
		char ch;
		if(!(is >>ch)){
			return ret;
		}
		if((ch != '\"') && (ch != '\'')){
			POSEIDON_LOG_WARNING("String open expected");
			is.setstate(std::ios::failbit);
			return ret;
		}
		const char equote = ch;
		enum {
			state_plain,
			state_escaped,
			state_utf_zero,
			state_utf_one,
			state_utf_two,
			state_utf_three,
		} state = state_plain;
		unsigned utf16_unit = 0;
		for(;;){
			if(!is.get(ch)){
				POSEIDON_LOG_WARNING("String not closed");
				is.setstate(std::ios::failbit);
				return ret;
			}
			if(state != state_plain){
				switch(state){
					unsigned hexc;

				case state_escaped:
					switch(ch){
					case '\"':
						ret += '\"';
						state = state_plain;
						break;
					case '\\':
						ret += '\\';
						state = state_plain;
						break;
					case '/':
						ret += '/';
						state = state_plain;
						break;
					case 'b':
						ret += '\b';
						state = state_plain;
						break;
					case 'f':
						ret += '\f';
						state = state_plain;
						break;
					case 'n':
						ret += '\n';
						state = state_plain;
						break;
					case 'r':
						ret += '\r';
						state = state_plain;
						break;
					case 'u':
						utf16_unit = 0;
						state = state_utf_zero;
						break;
					default:
						POSEIDON_LOG_WARNING("Unknown escaped character sequence");
						is.setstate(std::ios::failbit);
						return ret;
					}
					break;
				case state_utf_zero:
				case state_utf_one:
				case state_utf_two:
				case state_utf_three:
					hexc = (unsigned char)ch;
					if(('0' <= hexc) && (hexc <= '9')){
						hexc -= '0';
					} else if(('A' <= hexc) && (hexc <= 'F')){
						hexc -= 'A' - 0x0A;
					} else if(('a' <= hexc) && (hexc <= 'f')){
						hexc -= 'a' - 0x0A;
					} else {
						POSEIDON_LOG_WARNING("Invalid hex digit for \\u");
						is.setstate(std::ios::failbit);
						return ret;
					}
					utf16_unit <<= 4;
					utf16_unit |= hexc;
					switch(state){
					case state_utf_zero:
						state = state_utf_one;
						break;
					case state_utf_one:
						state = state_utf_two;
						break;
					case state_utf_two:
						state = state_utf_three;
						break;
					case state_utf_three:
						if(utf16_unit < 0x80){
							ret += static_cast<char>(utf16_unit);
						} else if(utf16_unit < 0x800){
							ret += static_cast<char>((utf16_unit >> 6) | 0xC0);
							ret += static_cast<char>((utf16_unit & 0x3F) | 0x80);
						} else {
							ret += static_cast<char>((utf16_unit >> 12) | 0xE0);
							ret += static_cast<char>(((utf16_unit >> 6) & 0x3F) | 0x80);
							ret += static_cast<char>((utf16_unit & 0x3F) | 0x80);
						}
						state = state_plain;
						break;
					default:
						std::terminate();
					}
					break;
				default:
					std::terminate();
				}
			} else if(ch == '\\'){
				state = state_escaped;
			} else if(ch == equote){
				break;
			} else {
				ret += ch;
				// state = state_plain;
			}
		}
		return ret;
	}
	double accept_number(std::istream &is){
		POSEIDON_PROFILE_ME;

		double ret = 0;
		if(!(is >>ret)){
			POSEIDON_LOG_WARNING("Number expected");
			is.setstate(std::ios::failbit);
			return ret;
		}
		return ret;
	}
	Json_object accept_object(std::istream &is){
		POSEIDON_PROFILE_ME;

		Json_object ret;
		char ch;
		if(!(is >>ch)){
			return ret;
		}
		if(ch != '{'){
			POSEIDON_LOG_WARNING("Object open expected");
			is.setstate(std::ios::failbit);
			return ret;
		}
		for(;;){
			if(!(is >>ch)){
				POSEIDON_LOG_WARNING("Object not closed");
				is.setstate(std::ios::failbit);
				return ret;
			}
			if(ch == '}'){
				break;
			}
			if(ch == ','){
				continue;
			}
			is.putback(ch);
			std::string name = accept_string(is);
			if(!(is >>ch)){
				return ret;
			}
			if(ch != ':'){
				POSEIDON_LOG_WARNING("Colon expected");
				is.setstate(std::ios::failbit);
				return ret;
			}
			ret.set(Rcnts(name), accept_element(is));
		}
		return ret;
	}
	Json_array accept_array(std::istream &is){
		POSEIDON_PROFILE_ME;

		Json_array ret;
		char ch;
		if(!(is >>ch)){
			return ret;
		}
		if(ch != '['){
			POSEIDON_LOG_WARNING("Array open expected");
			is.setstate(std::ios::failbit);
			return ret;
		}
		for(;;){
			if(!(is >>ch)){
				POSEIDON_LOG_WARNING("Array not closed");
				is.setstate(std::ios::failbit);
				return ret;
			}
			if(ch == ']'){
				break;
			}
			if(ch == ','){
				continue;
			}
			is.putback(ch);
			ret.push_back(accept_element(is));
		}
		return ret;
	}
	bool accept_boolean(std::istream &is){
		POSEIDON_PROFILE_ME;

		char ch;
		if(!(is >>ch)){
			return false;
		}
		if((ch != 'f') && (ch != 't')){
			POSEIDON_LOG_WARNING("Boolean expected");
			is.setstate(std::ios::failbit);
			return false;
		}
		char str[8];
		if((ch == 'f') && (is.readsome(str, 4) == 4) && (std::memcmp(str, "alse", 4) == 0)){
			return false;
		} else if((is.readsome(str, 3) == 3) && (std::memcmp(str, "rue", 3) == 0)){
			return true;
		} else {
			POSEIDON_LOG_WARNING("Boolean expected");
			is.setstate(std::ios::failbit);
			return false;
		}
	}
	Json_null accept_null(std::istream &is){
		POSEIDON_PROFILE_ME;

		char ch;
		if(!(is >>ch)){
			return NULLPTR;
		}
		if(ch != 'n'){
			POSEIDON_LOG_WARNING("Boolean expected");
			is.setstate(std::ios::failbit);
			return NULLPTR;
		}
		char str[8];
		if((is.readsome(str, 3) == 3) && (std::memcmp(str, "ull", 3) == 0)){
			return NULLPTR;
		} else {
			POSEIDON_LOG_WARNING("Boolean expected");
			is.setstate(std::ios::failbit);
			return NULLPTR;
		}
	}

	Json_element accept_element(std::istream &is){
		POSEIDON_PROFILE_ME;

		Json_element ret;
		char ch;
		if(!(is >>ch)){
			POSEIDON_LOG_WARNING("No input character");
			is.setstate(std::ios::failbit);
			return ret;
		}
		is.putback(ch);
		switch(ch){
		case '\"':
		case '\'':
			ret = accept_string(is);
			break;
		case '-':
		case '0': case '1': case '2': case '3': case '4':
		case '5': case '6': case '7': case '8': case '9':
			ret = accept_number(is);
			break;
		case '{':
			ret = accept_object(is);
			break;
		case '[':
			ret = accept_array(is);
			break;
		case 't': case 'f':
			ret = accept_boolean(is);
			break;
		case 'n':
			ret = accept_null(is);
			break;
		default:
			POSEIDON_LOG_WARNING("Unknown element type");
			is.setstate(std::ios::failbit);
			return ret;
		}
		return ret;
	}
}

const Json_element &null_json_element() NOEXCEPT {
	return g_null_element;
}

Json_object::Json_object(std::istream &is)
	: m_elements()
{
	parse(is);
	POSEIDON_THROW_UNLESS(is, Exception, Rcnts::view("Json_object parser error"));
}

Stream_buffer Json_object::dump() const {
	POSEIDON_PROFILE_ME;

	Buffer_ostream bos;
	dump(bos);
	return STD_MOVE(bos.get_buffer());
}
void Json_object::dump(std::ostream &os) const {
	POSEIDON_PROFILE_ME;

	os <<'{';
	AUTO(it, begin());
	if(it != end()){
		goto _loop_entry;
		do {
			os <<',';
	_loop_entry:
			os <<'\"';
			os <<it->first.get();
			os <<'\"';
			os <<':';
			os <<it->second;
		} while(++it != end());
	}
	os <<'}';
}
void Json_object::parse(std::istream &is){
	POSEIDON_PROFILE_ME;

	AUTO(obj, accept_object(is));
	if(is){
		obj.swap(*this);
	}
}

Json_array::Json_array(std::istream &is)
	: m_elements()
{
	parse(is);
	POSEIDON_THROW_UNLESS(is, Exception, Rcnts::view("Json_array parser error"));
}

Stream_buffer Json_array::dump() const {
	POSEIDON_PROFILE_ME;

	Buffer_ostream bos;
	dump(bos);
	return STD_MOVE(bos.get_buffer());
}
void Json_array::dump(std::ostream &os) const {
	POSEIDON_PROFILE_ME;

	os <<'[';
	AUTO(it, begin());
	if(it != end()){
		goto _loop_entry;
		do {
			os <<',';
	_loop_entry:
			os <<*it;
		} while(++it != end());
	}
	os <<']';
}
void Json_array::parse(std::istream &is){
	POSEIDON_PROFILE_ME;

	AUTO(arr, accept_array(is));
	if(is){
		arr.swap(*this);
	}
}

const char *Json_element::get_type_string(Json_element::Type type){
	switch(type){
	case type_boolean:
		return "boolean";
	case type_number:
		return "number";
	case type_string:
		return "string";
	case type_object:
		return "object";
	case type_array:
		return "array";
	case type_null:
		return "null";
	default:
		POSEIDON_LOG_WARNING("Unknown JSON element type: type = ", static_cast<int>(type));
		return "undefined";
	}
}

Stream_buffer Json_element::dump() const {
	POSEIDON_PROFILE_ME;

	Buffer_ostream bos;
	dump(bos);
	return STD_MOVE(bos.get_buffer());
}
void Json_element::dump(std::ostream &os) const {
	POSEIDON_PROFILE_ME;

	const Type type = get_type();
	switch(type){
	case type_boolean:
		os <<(get<bool>() ? "true" : "false");
		break;
	case type_number:
		os <<std::setprecision(16) <<get<double>();
		break;
	case type_string: {
		const AUTO_REF(str, get<std::string>());
		os <<'\"';
		for(AUTO(it, str.begin()); it != str.end(); ++it){
			const unsigned ch = (unsigned char)*it;
			switch(ch){
			case '\b':
				os <<'\\' <<'b';
				break;
			case '\f':
				os <<'\\' <<'f';
				break;
			case '\n':
				os <<'\\' <<'n';
				break;
			case '\r':
				os <<'\\' <<'r';
				break;
			case '\t':
				os <<'\\' <<'t';
				break;
			case '\"':
			case '\\':
			case '/':
				os <<'\\' <<(char)ch;
				break;
			default:
				if((ch < 0x20) || (ch == 0x7F) || (ch == 0xFF)){
					os <<'\\' <<'u' <<std::setfill('0') <<std::setw(4) <<ch;
				} else {
					os <<(char)ch;
				}
				break;
			}
		}
		os <<'\"';
		break; }
	case type_object:
		os <<get<Json_object>();
		break;
	case type_array:
		os <<get<Json_array>();
		break;
	case type_null:
		os <<"null";
		break;
	default:
		POSEIDON_LOG_FATAL("Unknown JSON element type: type = ", static_cast<int>(type));
		std::terminate();
	}
}
void Json_element::parse(std::istream &is){
	POSEIDON_PROFILE_ME;

	AUTO(elem, accept_element(is));
	if(is){
		elem.swap(*this);
	}
}

}
