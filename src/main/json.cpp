// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "json.hpp"
#include "protocol_exception.hpp"
#include <iomanip>

namespace Poseidon {

void JsonObject::dump(std::string &str) const {
	std::ostringstream oss;
	dump(oss);
	str = oss.str();
}
void JsonObject::dump(std::ostream &os) const {
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

void JsonArray::dump(std::string &str) const {
	std::ostringstream oss;
	dump(oss);
	str = oss.str();
}
void JsonArray::dump(std::ostream &os) const {
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

void JsonElement::dump(std::string &str) const {
	std::ostringstream oss;
	dump(oss);
	str = oss.str();
}
void JsonElement::dump(std::ostream &os) const {
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

std::string JsonParser::acceptString(std::istream &is){
	std::string ret;
	char temp;
	if(!(is >>temp) || (temp != '\"')){
		DEBUG_THROW(ProtocolException,
			SharedNts::observe("Bad JSON: expecting string open"), -1);
	}

	is >>std::noskipws;
	for(;;){
		if(!(is >>temp)){
			DEBUG_THROW(ProtocolException,
				SharedNts::observe("Bad JSON: expecting string close"), -1);
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
			DEBUG_THROW(ProtocolException,
				SharedNts::observe("Bad JSON: escape character at end"), -1);
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
				DEBUG_THROW(ProtocolException,
					SharedNts::observe("Bad JSON: too few hex digits for \\u"), -1);
			} else {
				unsigned code = 0;
				for(unsigned i = 12; i != (unsigned)-4; i -= 4){
					unsigned ch = (unsigned char)*read;
					++read;
					if(('0' <= ch) && (ch <= '9')){
						ch -= '0';
					} else if(('A' <= ch) && (ch <= 'F')){
						ch -= 'A' - 0x0A;
					} else if(('a' <= ch) && (ch <= 'f')){
						ch -= 'a' - 0x0A;
					} else {
						DEBUG_THROW(ProtocolException,
							SharedNts::observe("Bad JSON: invalid hex digit"), -1);
					}
					code |= ch << i;
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
			DEBUG_THROW(ProtocolException,
				SharedNts::observe("Bad JSON: invalid escape character"), -1);
		}
	}
	ret.erase(write, ret.end());
	return ret;
}
double JsonParser::acceptNumber(std::istream &is){
	double ret;
	if(is >>ret){
		return ret;
	}
	DEBUG_THROW(ProtocolException,
		SharedNts::observe("Bad JSON: expecting number"), -1);
}
JsonObject JsonParser::acceptObject(std::istream &is){
	JsonObject ret;
	char temp;
	if(!(is >>temp) || (temp != '{')){
		DEBUG_THROW(ProtocolException,
			SharedNts::observe("Bad JSON: expecting object open"), -1);
	}
	for(;;){
		if(!(is >>temp)){
			DEBUG_THROW(ProtocolException,
				SharedNts::observe("Bad JSON: end of stream"), -1);
		}
		if(temp == '}'){
			break;
		}
		if(temp == ','){
			continue;
		}
		is.unget();
		std::string name = acceptString(is);
		if(!(is >>temp) || (temp != ':')){
			DEBUG_THROW(ProtocolException,
				SharedNts::observe("Bad JSON: expecting colon"), -1);
		}
		JsonElement element = parseElement(is);
		ret[SharedNts(name)] = STD_MOVE(element);
	}
	return ret;
}
JsonArray JsonParser::acceptArray(std::istream &is){
	JsonArray ret;
	char temp;
	if(!(is >>temp) || (temp != '[')){
		DEBUG_THROW(ProtocolException,
			SharedNts::observe("Bad JSON: expecting array open"), -1);
	}
	for(;;){
		if(!(is >>temp)){
			DEBUG_THROW(ProtocolException,
				SharedNts::observe("Bad JSON: end of stream"), -1);
		}
		if(temp == ']'){
			break;
		}
		if(temp == ','){
			continue;
		}
		is.unget();
		JsonElement element = parseElement(is);
		ret.push_back(STD_MOVE(element));
	}
	return ret;
}
bool JsonParser::acceptBoolean(std::istream &is){
	bool ret;
	if(!(is >>std::boolalpha >>ret)){
		DEBUG_THROW(ProtocolException,
			SharedNts::observe("Bad JSON: expecting boolean"), -1);
	}
	return ret;
}
JsonNull JsonParser::acceptNull(std::istream &is){
	char temp[5];
	if(!(is >>std::setw(sizeof(temp)) >>temp) || (std::strcmp(temp, "null") != 0)){
		DEBUG_THROW(ProtocolException,
			SharedNts::observe("Bad JSON: expecting null"), -1);
	}
	return JsonNull();
}

JsonElement JsonParser::parseElement(std::istream &is){
	JsonElement ret;
	char temp;
	if(!(is >>temp)){
		DEBUG_THROW(ProtocolException,
			SharedNts::observe("Bad JSON: end of stream"), -1);
	}
	is.unget();
	switch(temp){
	case '\"':
		ret = acceptString(is);
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
		ret = acceptNumber(is);
		break;

	case '{':
		ret = acceptObject(is);
		break;

	case '[':
		ret = acceptArray(is);
		break;

	case 't':
	case 'f':
		ret = acceptBoolean(is);
		break;

	case 'n':
		ret = acceptNull(is);
		break;

	default:
		DEBUG_THROW(ProtocolException,
			SharedNts::observe("Bad JSON: unknown element type"), -1);
	}
	return ret;
}
JsonObject JsonParser::parseObject(std::istream &is){
	return acceptObject(is >>std::skipws);
}
JsonArray JsonParser::parseArray(std::istream &is){
	return acceptArray(is >>std::skipws);
}

}
