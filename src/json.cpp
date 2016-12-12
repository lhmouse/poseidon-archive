// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "json.hpp"
#include "log.hpp"
#include "profiler.hpp"
#include "buffer_streams.hpp"
#include <iomanip>

namespace Poseidon {

namespace {
	const JsonElement g_null_element = JsonNull();

	extern JsonElement accept_element(std::istream &is);

	std::string accept_string(std::istream &is){
		PROFILE_ME;

		std::string ret;
		char ch;
		if(!(is >>ch) || (ch != '\"')){
			LOG_POSEIDON_WARNING("String open expected");
			is.setstate(std::ios::badbit);
			return ret;
		}
		enum {
			S_PLAIN,
			S_ESCAPED,
			S_UTF_ZERO,
			S_UTF_ONE,
			S_UTF_TWO,
			S_UTF_THREE,
		} state = S_PLAIN;
		unsigned utf16_unit = 0;
		for(;;){
			if(!is.get(ch)){
				LOG_POSEIDON_WARNING("String not closed");
				is.setstate(std::ios::badbit);
				return ret;
			}
			if(state != S_PLAIN){
				switch(state){
					unsigned hexc;

				case S_ESCAPED:
					switch(ch){
					case '\"':
						ret += '\"';
						state = S_PLAIN;
						break;
					case '\\':
						ret += '\\';
						state = S_PLAIN;
						break;
					case '/':
						ret += '/';
						state = S_PLAIN;
						break;
					case 'b':
						ret += '\b';
						state = S_PLAIN;
						break;
					case 'f':
						ret += '\f';
						state = S_PLAIN;
						break;
					case 'n':
						ret += '\n';
						state = S_PLAIN;
						break;
					case 'r':
						ret += '\r';
						state = S_PLAIN;
						break;
					case 'u':
						utf16_unit = 0;
						state = S_UTF_ZERO;
						break;
					default:
						LOG_POSEIDON_WARNING("Unknown escaped character sequence");
						is.setstate(std::ios::badbit);
						return ret;
					}
					break;
				case S_UTF_ZERO:
				case S_UTF_ONE:
				case S_UTF_TWO:
				case S_UTF_THREE:
					hexc = (unsigned char)ch;
					if(('0' <= hexc) && (hexc <= '9')){
						hexc -= '0';
					} else if(('A' <= hexc) && (hexc <= 'F')){
						hexc -= 'A' - 0x0A;
					} else if(('a' <= hexc) && (hexc <= 'f')){
						hexc -= 'a' - 0x0A;
					} else {
						LOG_POSEIDON_WARNING("Invalid hex digit for \\u");
						is.setstate(std::ios::badbit);
						return ret;
					}
					utf16_unit <<= 4;
					utf16_unit |= hexc;
					switch(state){
					case S_UTF_ZERO:
						state = S_UTF_ONE;
						break;
					case S_UTF_ONE:
						state = S_UTF_TWO;
						break;
					case S_UTF_TWO:
						state = S_UTF_THREE;
						break;
					case S_UTF_THREE:
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
						state = S_PLAIN;
						break;
					default:
						std::abort();
					}
					break;
				default:
					std::abort();
				}
			} else if(ch == '\\'){
				state = S_ESCAPED;
			} else if(ch == '\"'){
				break;
			} else {
				ret += ch;
				// state = S_PLAIN;
			}
		}
		return ret;
	}
	double accept_number(std::istream &is){
		PROFILE_ME;

		double ret = 0;
		if(!(is >>ret)){
			LOG_POSEIDON_WARNING("Number expected");
			is.setstate(std::ios::badbit);
			return ret;
		}
		return ret;
	}
	JsonObject accept_object(std::istream &is){
		PROFILE_ME;

		JsonObject ret;
		char ch;
		if(!(is >>ch) || (ch != '{')){
			LOG_POSEIDON_WARNING("Object open expected");
			is.setstate(std::ios::badbit);
			return ret;
		}
		for(;;){
			if(!(is >>ch)){
				LOG_POSEIDON_WARNING("Object not closed");
				is.setstate(std::ios::badbit);
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
			if(!(is >>ch) || (ch != ':')){
				LOG_POSEIDON_WARNING("Colon expected");
				is.setstate(std::ios::badbit);
				return ret;
			}
			ret.set(SharedNts(name), accept_element(is));
		}
		return ret;
	}
	JsonArray accept_array(std::istream &is){
		PROFILE_ME;

		JsonArray ret;
		char ch;
		if(!(is >>ch) || (ch != '[')){
			LOG_POSEIDON_WARNING("Array open expected");
			is.setstate(std::ios::badbit);
			return ret;
		}
		for(;;){
			if(!(is >>ch)){
				LOG_POSEIDON_WARNING("Array not closed");
				is.setstate(std::ios::badbit);
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
		PROFILE_ME;

		char ch;
		if(!(is >>ch) || ((ch != 'f') && (ch != 't'))){
			LOG_POSEIDON_WARNING("Boolean expected");
			is.setstate(std::ios::badbit);
			return false;
		}
		char str[8];
		if((ch == 'f') && (is.readsome(str, 4) == 4) && (std::memcmp(str, "alse", 4) == 0)){
			return false;
		} else if((is.readsome(str, 3) == 3) && (std::memcmp(str, "rue", 3) == 0)){
			return true;
		} else {
			LOG_POSEIDON_WARNING("Boolean expected");
			is.setstate(std::ios::badbit);
			return false;
		}
	}
	JsonNull accept_null(std::istream &is){
		PROFILE_ME;

		char ch;
		if(!(is >>ch) || (ch != 'n')){
			LOG_POSEIDON_WARNING("Boolean expected");
			is.setstate(std::ios::badbit);
			return NULLPTR;
		}
		char str[8];
		if((is.readsome(str, 3) == 3) && (std::memcmp(str, "ull", 3) == 0)){
			return NULLPTR;
		} else {
			LOG_POSEIDON_WARNING("Boolean expected");
			is.setstate(std::ios::badbit);
			return NULLPTR;
		}
	}

	JsonElement accept_element(std::istream &is){
		PROFILE_ME;

		JsonElement ret;
		char ch;
		if(!(is >>ch)){
			LOG_POSEIDON_WARNING("No input character");
			is.setstate(std::ios::badbit);
			return ret;
		}
		is.putback(ch);
		switch(ch){
		case '\"':
			ret = accept_string(is);
			break;
		case '-':
		case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
			ret = accept_number(is);
			break;
		case '{':
			ret = accept_object(is);
			break;
		case '[':
			ret = accept_array(is);
			break;
		case 't':
		case 'f':
			ret = accept_boolean(is);
			break;
		case 'n':
			ret = accept_null(is);
			break;
		default:
			LOG_POSEIDON_WARNING("Unknown element type");
			is.setstate(std::ios::badbit);
			return ret;
		}
		return ret;
	}
}

const JsonElement &null_json_element() NOEXCEPT {
	return g_null_element;
}

std::string JsonObject::dump() const {
	PROFILE_ME;

	Buffer_ostream os;
	dump(os);
	return os.get_buffer().dump_string();
}
void JsonObject::dump(std::ostream &os) const {
	PROFILE_ME;

	os <<'{';
	AUTO(it, begin());
	if(it != end()){
		os <<'\"';
		os <<it->first.get();
		os <<'\"';
		os <<':';
		it->second.dump(os);

		while(++it != end()){
			os <<',';

			os <<'\"';
			os <<it->first.get();
			os <<'\"';
			os <<':';
			it->second.dump(os);
		}
	}
	os <<'}';
}
void JsonObject::parse(std::istream &is){
	accept_object(is).swap(*this);
}

std::string JsonArray::dump() const {
	PROFILE_ME;

	Buffer_ostream os;
	dump(os);
	return os.get_buffer().dump_string();
}
void JsonArray::dump(std::ostream &os) const {
	PROFILE_ME;

	os <<'[';
	AUTO(it, begin());
	if(it != end()){
		it->dump(os);

		while(++it != end()){
			os <<',';

			it->dump(os);
		}
	}
	os <<']';
}
void JsonArray::parse(std::istream &is){
	PROFILE_ME;

	accept_array(is).swap(*this);
}

const char *JsonElement::get_type_string(JsonElement::Type type){
	switch(type){
	case T_BOOL:
		return "Boolean";
	case T_NUMBER:
		return "Number";
	case T_STRING:
		return "String";
	case T_OBJECT:
		return "Object";
	case T_ARRAY:
		return "Array";
	case T_NULL:
		return "Null";
	default:
		LOG_POSEIDON_WARNING("Unknown JSON element type: type = ", static_cast<int>(type));
		return "Undefined";
	}
}

std::string JsonElement::dump() const {
	PROFILE_ME;

	Buffer_ostream os;
	dump(os);
	return os.get_buffer().dump_string();
}
void JsonElement::dump(std::ostream &os) const {
	PROFILE_ME;

	const Type type = get_type();
	switch(type){
	case T_BOOL:
		os <<std::boolalpha <<get<bool>();
		break;
	case T_NUMBER:
		os <<std::setprecision(20) <<get<double>();
		break;
	case T_STRING: {
		const AUTO_REF(str, get<std::string>());
		os <<'\"';
		for(AUTO(it, str.begin()); it != str.end(); ++it){
			const unsigned ch = (unsigned char)*it;
			switch(ch){
			case '\"':
			case '\\':
			case '/':
				os <<'\\' <<*it;
				break;
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
		} break;
	case T_OBJECT:
		os <<get<JsonObject>();
		break;
	case T_ARRAY:
		os <<get<JsonArray>();
		break;
	case T_NULL:
		os <<"null";
		break;
	default:
		LOG_POSEIDON_FATAL("Unknown JSON element type: type = ", static_cast<int>(type));
		std::abort();
	}
}
void JsonElement::parse(std::istream &is){
	accept_element(is).swap(*this);
}

}
