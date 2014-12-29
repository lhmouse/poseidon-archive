// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "json_writer.hpp"
#include <iomanip>
using namespace Poseidon;

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
std::string JsonObject::dump() const {
	std::ostringstream os;
	dump(os);
	return os.str();
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
std::string JsonArray::dump() const {
	std::ostringstream os;
	dump(os);
	return os.str();
}

void JsonElement::dump(std::ostream &os) const {
	switch(m_data.which()){
	case 0: // bool
		{
			const bool *const p = boost::get<bool>(&m_data);
			assert(p);
			os <<(*p ? "true" : "false");
		}
		break;

	case 1: // long double
		{
			const long double *const p = boost::get<long double>(&m_data);
			assert(p);
			os <<std::setprecision(20) <<*p;
		}
		break;

	case 2: // std::string
		{
			const std::string *const p = boost::get<std::string>(&m_data);
			assert(p);
			os <<'\"';
			for(AUTO(it, p->begin()); it != p->end(); ++it){
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

	case 3: // JsonObject
		{
			const JsonObject *const p = boost::get<JsonObject>(&m_data);
			assert(p);
			p->dump(os);
		}
		break;

	case 4: // JsonArray
		{
			const JsonArray *const p = boost::get<JsonArray>(&m_data);
			assert(p);
			p->dump(os);
		}
		break;

	case 5: // JsonNull
		{
			os <<"null";
		}
		break;

	default:
		assert(false);
	}
}
std::string JsonElement::dump() const {
	std::ostringstream os;
	dump(os);
	return os.str();
}
