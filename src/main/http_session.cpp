#include "../precompiled.hpp"
#include "http_session.hpp"
#include "http_status.hpp"
#include "log.hpp"
using namespace Poseidon;

namespace {

void respond(HttpSession &session, HttpStatus status,
	OptionalMap &headers, std::string &contents)
{
	std::ostringstream ss;
	std::string str;

	const AUTO(desc, getHttpStatusCodeDesc(status));
	if(contents.empty() && ((unsigned)status / 100 != 2)){
		ss <<"<html><head><title>" <<(unsigned)status <<' ' <<desc.descShort
			<<"</title></head><body><h1>" <<(unsigned)status <<' ' <<desc.descShort
			<<"</h1><hr /><p>" <<desc.descLong <<"</p></body></html>";
		contents = ss.str();
		headers.set("Content-Length", boost::lexical_cast<std::string>(contents.size()));
		headers.set("Content-Type", "text/html");
	} else {
		AUTO_REF(contentType, headers["Content-Type"]);
		if(contentType.empty()){
			contentType.assign("text/plain; charset=utf-8");
		}
	}

	ss.str("");
	ss <<"HTTP/1.1 " <<(unsigned)status <<' ' <<desc.descShort <<"\r\n";
	for(AUTO(it, headers.begin()); it != headers.end(); ++it){
		ss <<it->first.get() <<": " <<it->second <<"\r\n";
	}
	ss <<"\r\n";
	ss <<contents;

	str = ss.str();
	session.send(str.data(), str.size());
}

}

HttpSession::HttpSession(ScopedFile &socket)
	: TcpSessionBase(socket), m_state(ST_FIRST_HEADER)
{
}

void HttpSession::onReadAvail(const void *data, std::size_t size){
	LOG_DEBUG("Received ", std::string((const char *)data, size));

	OptionalMap headers;
	std::string contents;
	respond(*this, HTTP_NOT_FOUND, headers, contents);
}
void HttpSession::onRemoteClose(){
}

void HttpSession::perform() const {
}
