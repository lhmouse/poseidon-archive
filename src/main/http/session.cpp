#include "../../precompiled.hpp"
#include "session.hpp"
#include <string.h>
#include "request.hpp"
#include "exception.hpp"
#include "utilities.hpp"
#include "upgraded_session_base.hpp"
#include "websocket/session.hpp"
#include "../log.hpp"
#include "../singletons/http_servlet_manager.hpp"
#include "../singletons/websocket_servlet_manager.hpp"
#include "../singletons/timer_daemon.hpp"
#include "../stream_buffer.hpp"
#include "../utilities.hpp"
#include "../exception.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../hash.hpp"
using namespace Poseidon;

namespace {

StreamBuffer makeResponse(HttpStatus status, unsigned version,
	OptionalMap headers = OptionalMap(), StreamBuffer *contents = NULLPTR)
{
	LOG_DEBUG("Making HTTP response: status = ", (unsigned)status);

	char codeStatus[512];
	std::size_t codeStatusLen = std::sprintf(codeStatus, "%u ", (unsigned)status);
	const AUTO(desc, getHttpStatusDesc(status));
	const std::size_t toAppend = std::min(
		sizeof(codeStatus) - codeStatusLen, std::strlen(desc.descShort));
	std::memcpy(codeStatus + codeStatusLen, desc.descShort, toAppend);
	codeStatusLen += toAppend;

	StreamBuffer realContents;
	if(!contents){
		realContents.put("<html><head><title>");
		realContents.put(codeStatus, codeStatusLen);
		realContents.put("</title></head><body><h1>");
		realContents.put(codeStatus, codeStatusLen);
		realContents.put("</h1><hr /><p>");
		realContents.put(desc.descLong);
		realContents.put("</p></body></html>");

		headers.set("Content-Type", "text/html; charset=utf-8");
		headers.set("Content-Length", boost::lexical_cast<std::string>(realContents.size()));
	} else if(!contents->empty()){
		realContents.splice(*contents);

		AUTO_REF(contentType, headers.create("Content-Type"));
		if(contentType.empty()){
			contentType.assign("text/plain; charset=utf-8");
		}
		headers.set("Content-Length", boost::lexical_cast<std::string>(realContents.size()));
	}

	char versionStr[64];
	const std::size_t versionStrLen = std::sprintf(
		versionStr, "HTTP/%u.%u ", version / 10000, version % 10000);
	StreamBuffer ret;
	ret.put(versionStr, versionStrLen);
	ret.put(codeStatus, codeStatusLen);
	ret.put("\r\n");
	for(AUTO(it, headers.begin()); it != headers.end(); ++it){
		if(!it->second.empty()){
			ret.put(it->first.get(), std::strlen(it->first.get()));
			ret.put(": ");
			ret.put(it->second.data(), it->second.size());
			ret.put("\r\n");
		}
	}
	ret.put("\r\n");
	ret.splice(realContents);
	return ret;
}

void normalizeUri(std::string &uri){
	if(uri[0] != '/'){
		uri.insert(uri.begin(), '/');
	}
	std::size_t write = 1;
	for(std::size_t i = 1; i < uri.size(); ++i){
		const char ch = uri[i];
		if((ch == '/') && (uri[write - 1] == '/')){
			continue;
		}
		uri[write] = ch;
		++write;
	}
	uri.erase(uri.begin() + write, uri.end());
}

void onRequestTimeout(const boost::weak_ptr<HttpSession> &observer, unsigned long long){
	const AUTO(session, observer.lock());
	if(session){
		LOG_WARNING("HTTP request times out, remote IP = ", session->getRemoteIp());
		session->shutdown(HTTP_REQUEST_TIMEOUT);
	}
}
void onKeepAliveTimeout(const boost::weak_ptr<HttpSession> &observer, unsigned long long){
	const AUTO(session, observer.lock());
	if(session){
		session->TcpSessionBase::shutdown();
	}
}

class HttpRequestJob : public JobBase {
private:
	const boost::weak_ptr<HttpSession> m_session;

	HttpRequest m_request;

public:
	HttpRequestJob(boost::weak_ptr<HttpSession> session,
		HttpVerb verb, std::string uri, unsigned version,
		OptionalMap getParams, OptionalMap headers, std::string contents)
		: m_session(STD_MOVE(session))
	{
		m_request.verb = verb;
		m_request.uri.swap(uri);
		m_request.version = version;
		m_request.getParams.swap(getParams);
		m_request.headers.swap(headers);
		m_request.contents.swap(contents);
	}

protected:
	void perform(){
		assert(!m_request.uri.empty());

		const AUTO(session, m_session.lock());
		if(!session){
			LOG_WARNING("The specified HTTP session has expired.");
			return;
		}
		const unsigned version = m_request.version;
		try {
			boost::shared_ptr<const void> lockedDep;
			AUTO(servlet, HttpServletManager::getServlet(lockedDep, m_request.uri));
			if(!servlet){
				LOG_WARNING("No handler matches URI ", m_request.uri);
				DEBUG_THROW(HttpException, HTTP_NOT_FOUND);
			}
			LOG_DEBUG("Dispatching http request: URI = ", m_request.uri,
				", verb = ", stringFromHttpVerb(m_request.verb));
			const HttpVerb verb = m_request.verb;
			OptionalMap headers;
			StreamBuffer contents;
			const HttpStatus status = (*servlet)(headers, contents, STD_MOVE(m_request));
			if((verb == HTTP_HEAD) || (status == HTTP_NO_CONTENT) || ((unsigned)status / 100 == 1)){
				contents.clear();
			}
			session->send(makeResponse(status, version, STD_MOVE(headers), &contents));
		} catch(HttpException &e){
			LOG_ERROR("HttpException thrown in HTTP servlet, status = ", e.status(),
				", file = ", e.file(), ", line = ", e.line());
			session->send(makeResponse(e.status(), version));
			throw;
		} catch(...){
			LOG_ERROR("Forwarding exception... shutdown the session first.");
			session->send(makeResponse(HTTP_SERVER_ERROR, version));
			throw;
		}
	}
};

}

HttpSession::HttpSession(Move<ScopedFile> socket)
	: TcpSessionBase(STD_MOVE(socket))
	, m_state(ST_FIRST_HEADER), m_totalLength(0), m_contentLength(0)
	, m_verb(HTTP_GET), m_version(10000)
{
}
HttpSession::~HttpSession(){
	if(m_state != ST_FIRST_HEADER){
		LOG_WARNING("Now that this HTTP session is to be destroyed, "
			"a premature request has to be discarded.");
	}
}

void HttpSession::setRequestTimeout(unsigned long long timeout){
	LOG_DEBUG("Setting request timeout to ", timeout);

	if(timeout == 0){
		m_shutdownTimer.reset();
	} else {
		m_shutdownTimer = TimerDaemon::registerTimer(timeout, 0, NULLPTR,
			TR1::bind(&onRequestTimeout, virtualWeakFromThis<HttpSession>(), TR1::placeholders::_1));
	}
}

void HttpSession::onAllHeadersRead(){
	typedef std::pair<const char *, void (HttpSession::*)(const std::string &)> TableItem;

	static const TableItem JUMP_TABLE[] = {
		std::make_pair("Content-Length", &HttpSession::onContentLength),
		std::make_pair("Expect", &HttpSession::onExpect),
		std::make_pair("Upgrade", &HttpSession::onUpgrade),
	};

	for(AUTO(it, m_headers.begin()); it != m_headers.end(); ++it){
		AUTO(lower, BEGIN(JUMP_TABLE));
		AUTO(upper, END(JUMP_TABLE));
		void (HttpSession::*found)(const std::string &) = 0;
		do {
			const AUTO(middle, lower + (upper - lower) / 2);
			const int result = std::strcmp(it->first.get(), middle->first);
			if(result == 0){
				found = middle->second;
				break;
			} else if(result < 0){
				upper = middle;
			} else {
				lower = middle + 1;
			}
		} while(lower != upper);

		if(found){
			(this->*found)(it->second);
		}
	}
}
void HttpSession::onExpect(const std::string &val){
	if(val != "100-continue"){
		LOG_WARNING("Unknown HTTP header Expect: ", val);
		DEBUG_THROW(HttpException, HTTP_NOT_SUPPORTED);
	}
	send(makeResponse(HTTP_CONTINUE, m_version));
}
void HttpSession::onContentLength(const std::string &val){
	m_contentLength = boost::lexical_cast<std::size_t>(val);
	LOG_DEBUG("Content-Length: ", m_contentLength);
}
void HttpSession::onUpgrade(const std::string &val){
	if(m_version < 10001){
		LOG_WARNING("HTTP 1.1 is required to use WebSocket");
		DEBUG_THROW(HttpException, HTTP_NOT_SUPPORTED);
	}
	if(::strcasecmp(val.c_str(), "websocket") != 0){
		LOG_WARNING("Unknown HTTP header Upgrade: ", val);
		DEBUG_THROW(HttpException, HTTP_NOT_SUPPORTED);
	}
	AUTO_REF(version, m_headers.get("Sec-WebSocket-Version"));
	if(version != "13"){
		LOG_WARNING("Unknown HTTP header Sec-WebSocket-Version: ", version);
		DEBUG_THROW(HttpException, HTTP_NOT_SUPPORTED);
	}

	// 仅测试。
	boost::shared_ptr<const void> lockedDep;
	if(!WebSocketServletManager::getServlet(lockedDep, m_uri)){
		DEBUG_THROW(HttpException, HTTP_NOT_FOUND);
	}

	std::string key = m_headers.get("Sec-WebSocket-Key");
	if(key.empty()){
		LOG_WARNING("No Sec-WebSocket-Key specified.");
		DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
	}
	key += "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
	unsigned char sha1[20];
	sha1Sum(sha1, key.data(), key.size());
	key = base64Encode(sha1, sizeof(sha1));

	OptionalMap headers;
	headers.set("Upgrade", "websocket");
	headers.set("Connection", "Upgrade");
	headers.set("Sec-WebSocket-Accept", STD_MOVE(key));
	StreamBuffer contents;
	send(makeResponse(HTTP_SWITCH_PROTOCOLS, m_version, headers, &contents));

	m_upgradedSession = boost::make_shared<WebSocketSession>(virtualWeakFromThis<HttpSession>());
	LOG_INFO("Upgraded to WebSocketSession, remote IP = ", getRemoteIp());
}

void HttpSession::onReadAvail(const void *data, std::size_t size){
	PROFILE_ME;

	if((m_state == ST_FIRST_HEADER) && m_upgradedSession){
		return m_upgradedSession->onReadAvail(data, size);
	}

	AUTO(read, (const char *)data);
	const AUTO(end, read + size);
	try {
		const std::size_t maxRequestLength = HttpServletManager::getMaxRequestLength();
		if(m_totalLength + size >= maxRequestLength){
			LOG_WARNING("Request size is ", m_totalLength + size, ", max = ", maxRequestLength);
			DEBUG_THROW(HttpException, HTTP_REQUEST_TOO_LARGE);
		}
		m_totalLength += size;

		while(read != end){
			if(m_state != ST_CONTENTS){
				const char ch = *read;
				++read;
				if(ch != '\n'){
					m_line.push_back(ch);
					continue;
				}
				const std::size_t lineLen = m_line.size();
				if((lineLen > 0) && (m_line[lineLen - 1] == '\r')){
					m_line.erase(m_line.end() - 1, m_line.end());
				}
				if(m_state == ST_FIRST_HEADER){
					if(m_line.empty()){
						continue;
					}
					AUTO(parts, explode<std::string>(' ', m_line, 3));
					if(parts.size() != 3){
						LOG_WARNING("Bad HTTP header: ", m_line);
						DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
					}

					m_verb = httpVerbFromString(parts[0].c_str());
					if(m_verb == HTTP_INVALID_VERB){
						LOG_WARNING("Bad HTTP verb: ", parts[0]);
						DEBUG_THROW(HttpException, HTTP_BAD_METHOD);
					}

					if(parts[1].empty() || (parts[1][0] != '/')){
						LOG_WARNING("Bad HTTP request URI: ", parts[1]);
						DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
					}
					m_uri = STD_MOVE(parts[1]);
					std::size_t pos = m_uri.find('#');
					if(pos != std::string::npos){
						m_uri.erase(m_uri.begin() + pos, m_uri.end());
					}
					pos = m_uri.find('?');
					if(pos == std::string::npos){
						m_getParams.clear();
					} else {
						m_getParams = optionalMapFromUrlEncoded(m_uri.substr(pos + 1));
						m_uri.erase(m_uri.begin() + pos, m_uri.end());
					}
					normalizeUri(m_uri);

					char versionMajor[16];
					char versionMinor[16];
					const int result = std::sscanf(parts[2].c_str(),
						"HTTP/%15[0-9].%15[0-9]%*c", versionMajor, versionMinor);
					if(result != 2){
						LOG_WARNING("Bad protocol string: ", parts[2]);
						DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
					}
					m_version = std::atoi(versionMajor) * 10000 + std::atoi(versionMinor);
					if((m_version != 10000) && (m_version != 10001)){
						LOG_WARNING("Bad HTTP version: ", parts[2]);
						DEBUG_THROW(HttpException, HTTP_VERSION_NOT_SUP);
					}

					m_state = ST_HEADERS;
				} else if(!m_line.empty()){
					const std::size_t delimPos = m_line.find(':');
					if(delimPos == std::string::npos){
						LOG_WARNING("Bad HTTP header: ", m_line);
						DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
					}
					AUTO(valueBegin, m_line.begin() + delimPos + 1);
					while(*valueBegin == ' '){
						++valueBegin;
					}
					std::string value(valueBegin, m_line.end());
					m_line.erase(m_line.begin() + delimPos, m_line.end());
					m_headers.add(m_line, STD_MOVE(value));
				} else {
					m_state = ST_CONTENTS;

					onAllHeadersRead();
				}
			}
			if(m_state != ST_CONTENTS){
				m_line.clear();
				continue;
			}

			const std::size_t bytesAvail = (std::size_t)(end - read);
			const std::size_t bytesRemaining = m_contentLength - m_line.size();
			if(bytesAvail < bytesRemaining){
				m_line.append(read, bytesAvail);
				read += bytesAvail;
				continue;
			}

			m_line.append(read, bytesRemaining);
			read += bytesRemaining;
			m_state = ST_FIRST_HEADER;

			if(m_upgradedSession){
				m_shutdownTimer.reset();

				m_upgradedSession->onInitContents(m_line.data(), m_line.size());
			} else {
				m_shutdownTimer = TimerDaemon::registerTimer(
					HttpServletManager::getKeepAliveTimeout(), 0, NULLPTR,
					TR1::bind(&onKeepAliveTimeout,
						virtualWeakFromThis<HttpSession>(), TR1::placeholders::_1));

				boost::make_shared<HttpRequestJob>(virtualWeakFromThis<HttpSession>(),
					m_verb, STD_MOVE(m_uri), m_version,
					STD_MOVE(m_getParams), STD_MOVE(m_headers), STD_MOVE(m_line)
					)->pend();

				m_totalLength = 0;
				m_contentLength = 0;
				m_line.clear();

				m_verb = HTTP_GET;
				m_uri.clear();
				m_version = 10000;
				m_getParams.clear();
				m_headers.clear();
			}
		}
	} catch(HttpException &e){
		LOG_ERROR("HttpException thrown while parsing HTTP data, status = ", e.status(),
			", file = ", e.file(), ", line = ", e.line());
		shutdown(e.status());
		throw;
	} catch(...){
		LOG_ERROR("Forwarding exception... shutdown the session first.");
		shutdown(HTTP_SERVER_ERROR);
		throw;
	}
}
bool HttpSession::shutdown(HttpStatus status){
	return TcpSessionBase::shutdown(makeResponse(status, m_version));
}
bool HttpSession::shutdown(HttpStatus status, OptionalMap headers, StreamBuffer contents){
	return TcpSessionBase::shutdown(makeResponse(status, m_version, STD_MOVE(headers), &contents));
}
