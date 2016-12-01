// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "json.hpp"
#include "log.hpp"
#include "profiler.hpp"
#include <iomanip>

namespace Poseidon {

namespace {
	const JsonElement g_null_element = JsonNull();

	extern JsonElement accept_element(std::istream &is);

	std::string accept_string(std::istream &is){
		PROFILE_ME;

		std::string ret;
		char temp;
		if(!(is >>temp) || (temp != '\"')){
			LOG_POSEIDON_WARNING("String open expected");
			is.setstate(std::istream::badbit);
			return ret;
		}

		is >>std::noskipws;
		for(;;){
			if(!(is >>temp)){
				LOG_POSEIDON_WARNING("String not closed");
				is.setstate(std::istream::badbit);
				return ret;
			}
			if(temp == '\"'){
				if(ret.empty() || (*ret.rbegin() != '\\')){
					break;
				}
			}
			ret.push_back(temp);
		}
		is >>std::skipws;

		std::string::iterator write = ret.begin();
		std::string::const_iterator read = write;
		while(read != ret.end()){
			char ch = *read;
			++read;

			if(ch != '\\'){
				*write = ch;
				++write;
				continue;
			}

			if(read == ret.end()){
				LOG_POSEIDON_WARNING("Found escape character at the end");
				is.setstate(std::istream::badbit);
				return ret;
			}
			ch = *read;
			++read;

			switch(ch){
			case '\"':
				*write = '\"';
				++write;
				break;

			case '\\':
				*write = '\\';
				++write;
				break;

			case '/':
				*write = '/';
				++write;
				break;

			case 'b':
				*write = '\b';
				++write;
				break;

			case 'f':
				*write = '\f';
				++write;
				break;

			case 'n':
				*write = '\n';
				++write;
				break;

			case 'r':
				*write = '\r';
				++write;
				break;

			case 'u':
				if(ret.end() - read < 4){
					LOG_POSEIDON_WARNING("Too few hex digits for \\u");
					is.setstate(std::istream::badbit);
					return ret;
				} else {
					unsigned code = 0;
					for(unsigned i = 12; i != (unsigned)-4; i -= 4){
						unsigned hexc = (unsigned char)*read;
						++read;
						if(('0' <= hexc) && (hexc <= '9')){
							hexc -= '0';
						} else if(('A' <= hexc) && (hexc <= 'F')){
							hexc -= 'A' - 0x0A;
						} else if(('a' <= hexc) && (hexc <= 'f')){
							hexc -= 'a' - 0x0A;
						} else {
							LOG_POSEIDON_WARNING("Invalid hex digits for \\u");
							is.setstate(std::istream::badbit);
							return ret;
						}
						code |= hexc << i;
					}
					if(code <= 0x7F){
						*write = code;
						++write;
					} else if(code <= 0x7FF){
						*write = (code >> 6) | 0xC0;
						++write;
						*write = (code & 0x3F) | 0x80;
						++write;
					} else {
						*write = (code >> 12) | 0xE0;
						++write;
						*write = ((code >> 6) & 0x3F) | 0x80;
						++write;
						*write = (code & 0x3F) | 0x80;
						++write;
					}
				}
				break;

			default:
				LOG_POSEIDON_WARNING("Unknown escaped character sequence");
				is.setstate(std::istream::badbit);
				return ret;
			}
		}
		ret.erase(write, ret.end());
		return ret;
	}
	double accept_number(std::istream &is){
		PROFILE_ME;

		double ret = 0;
		if(!(is >>ret)){
			LOG_POSEIDON_WARNING("Number expected");
			is.setstate(std::istream::badbit);
			return ret;
		}
		return ret;
	}
	JsonObject accept_object(std::istream &is){
		PROFILE_ME;

		JsonObject ret;
		char temp;
		if(!(is >>temp) || (temp != '{')){
			LOG_POSEIDON_WARNING("Object open expected");
			is.setstate(std::istream::badbit);
			return ret;
		}
		for(;;){
			if(!(is >>temp)){
				LOG_POSEIDON_WARNING("Object not closed");
				is.setstate(std::istream::badbit);
				return ret;
			}
			if(temp == '}'){
				break;
			}
			if(temp == ','){
				continue;
			}
			is.unget();
			std::string name = accept_string(is);
			if(!(is >>temp) || (temp != ':')){
				LOG_POSEIDON_WARNING("Colon expected");
				is.setstate(std::istream::badbit);
				return ret;
			}
			ret.set(SharedNts(name), accept_element(is));
		}
		return ret;
	}
	JsonArray accept_array(std::istream &is){
		PROFILE_ME;

		JsonArray ret;
		char temp;
		if(!(is >>temp) || (temp != '[')){
			LOG_POSEIDON_WARNING("Array open expected");
			is.setstate(std::istream::badbit);
			return ret;
		}
		for(;;){
			if(!(is >>temp)){
				LOG_POSEIDON_WARNING("Array not closed");
				is.setstate(std::istream::badbit);
				return ret;
			}
			if(temp == ']'){
				break;
			}
			if(temp == ','){
				continue;
			}
			is.unget();
			ret.push_back(accept_element(is));
		}
		return ret;
	}
	bool accept_boolean(std::istream &is){
		PROFILE_ME;

		bool ret = false;
		if(!(is >>std::boolalpha >>ret)){
			LOG_POSEIDON_WARNING("Boolean expected");
			is.setstate(std::istream::badbit);
			return ret;
		}
		return ret;
	}
	JsonNull accept_null(std::istream &is){
		PROFILE_ME;

		JsonNull ret = NULLPTR;
		char temp[5];
		if(!(is >>std::setw(sizeof(temp)) >>temp) || (std::strcmp(temp, "null") != 0)){
			LOG_POSEIDON_WARNING("Null expected");
			is.setstate(std::istream::badbit);
			return ret;
		}
		return ret;
	}

	JsonElement accept_element(std::istream &is){
		PROFILE_ME;

		JsonElement ret;
		char temp;
		if(!(is >>temp)){
			LOG_POSEIDON_WARNING("No input character");
			is.setstate(std::istream::badbit);
			return ret;
		}
		is.unget();
		switch(temp){
		case '\"':
			ret = accept_string(is);
			break;

		case '-':
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
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
			is.setstate(std::istream::badbit);
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

	std::ostringstream oss;
	dump(oss);
	return oss.str();
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

	std::ostringstream oss;
	dump(oss);
	return oss.str();
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

std::string JsonElement::dump() const {
	PROFILE_ME;

	std::ostringstream oss;
	dump(oss);
	return oss.str();
}
void JsonElement::dump(std::ostream &os) const {
	PROFILE_ME;

	switch(type()){
	case T_BOOL:
		os <<std::boolalpha <<get<bool>();
		break;

	case T_NUMBER:
		os <<std::setprecision(20) <<get<double>();
		break;

	case T_STRING:
		{
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
		}
		break;

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
		assert(false);
	}
}
void JsonElement::parse(std::istream &is){
	accept_element(is).swap(*this);
}

}
