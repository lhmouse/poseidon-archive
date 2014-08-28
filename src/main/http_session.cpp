#include "../precompiled.hpp"
#include "http_session.hpp"
#include "http_status.hpp"
#include "log.hpp"
#include <stdio.h>	// C99 snprintf()
using namespace Poseidon;

namespace {

void respond(HttpSession &session, HttpStatus status, OptionalMap &headers, std::string &contents){
	const AUTO(desc, getHttpStatusCodeDesc(status));
	const AUTO(codeStatus, boost::lexical_cast<std::string>((unsigned)status) + ' ' + desc.descShort);

	if(contents.empty() && ((unsigned)status / 100 != 2)){
		contents = "<html><head><title>";
		contents += codeStatus;
		contents += "</title></head><body><h1>";
		contents += codeStatus;
		contents += "</h1><hr /><p>";
		contents += desc.descLong;
		contents += "</p></body></html>";

		headers.set("Content-Type", "text/html");
	} else {
		AUTO_REF(contentType, headers["Content-Type"]);
		if(contentType.empty()){
			contentType.assign("text/plain; charset=utf-8");
		}
	}
	headers.set("Content-Length", boost::lexical_cast<std::string>(contents.size()));

	session.send("HTTP/1.1 ", 9);
	session.send(codeStatus.data(), codeStatus.size());
	session.send("\r\n", 2);
	for(AUTO(it, headers.begin()); it != headers.end(); ++it){
		if(!it->second.empty()){
			session.send(it->first.get(), std::strlen(it->first.get()));
			session.send(": ", 2);
			session.send(it->second.data(), it->second.size());
			session.send("\r\n", 2);
		}
	}
	session.send("\r\n", 2);
	session.send(contents.data(), contents.size());
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
	respond(*this, HTTP_NOT_SUPPORTED, headers, contents);
}
void HttpSession::onRemoteClose(){
}

void HttpSession::perform() const {
}
