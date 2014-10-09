#include "../../precompiled.hpp"
#include "session.hpp"
#include "status.hpp"
#include "request.hpp"
#include "exception.hpp"
#include "utilities.hpp"
#include "../log.hpp"
#include "../singletons/http_servlet_manager.hpp"
#include "../singletons/config_file.hpp"
#include "../singletons/timer_daemon.hpp"
#include "../stream_buffer.hpp"
#include "../utilities.hpp"
#include "../exception.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
using namespace Poseidon;

namespace {

const unsigned long long MAX_REQUEST_LENGTH = 0x4000;	// 头部加正文总长度。

void respond(HttpSession *session, HttpStatus status,
	OptionalMap headers = OptionalMap(), StreamBuffer *contents = NULLPTR)
{
	LOG_DEBUG("Sending HTTP response: status = ", (unsigned)status);

	char codeStatus[512];
	std::size_t codeStatusLen = std::sprintf(codeStatus, "%u ", (unsigned)status);
	const AUTO(desc, getHttpStatusCodeDesc(status));
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

	StreamBuffer buffer;
	buffer.put("HTTP/1.1 ");
	buffer.put(codeStatus, codeStatusLen);
	buffer.put("\r\n");
	for(AUTO(it, headers.begin()); it != headers.end(); ++it){
		if(!it->second.empty()){
			buffer.put(it->first.get(), std::strlen(it->first.get()));
			buffer.put(": ");
			buffer.put(it->second.data(), it->second.size());
			buffer.put("\r\n");
		}
	}
	buffer.put("\r\n");
	buffer.splice(realContents);

	session->sendUsingMove(buffer);
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

unsigned long long getHttpKeepAliveTimeout(){
	static unsigned long long ret =
		ConfigFile::get<unsigned long long>("http_keep_alive_timeout", 15000);
	return ret;
}

void onSessionTimeout(const boost::weak_ptr<HttpSession> &observer, unsigned long long){
	const AUTO(session, observer.lock());
	if(session){
		LOG_WARNING("HTTP session times out, remote ip = ", session->getRemoteIp());
		respond(session.get(), HTTP_REQUEST_TIMEOUT);
		session->shutdown();
	}
}

class HttpRequestJob : public JobBase {
private:
	boost::weak_ptr<HttpSession> m_session;
	HttpRequest m_request;

	boost::shared_ptr<const void> m_lockedDep;

public:
	HttpRequestJob(boost::weak_ptr<HttpSession> session,
		HttpVerb verb, std::string uri, OptionalMap getParams,
		OptionalMap headers, std::string contents)
		: m_session(STD_MOVE(session))
	{
		m_request.verb = verb;
		m_request.uri.swap(uri);
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
		try {
			AUTO(servlet, HttpServletManager::getServlet(m_lockedDep, m_request.uri));
			if(!servlet){
				LOG_WARNING("No handler matches URI ", m_request.uri);
				DEBUG_THROW(HttpException, HTTP_NOT_FOUND);
			}
			LOG_DEBUG("Dispatching http request: URI = ", m_request.uri,
				", verb = ", stringFromHttpVerb(m_request.verb));
			OptionalMap headers;
			StreamBuffer contents;
			const HttpStatus status = (*servlet)(headers, contents, STD_MOVE(m_request));
			if((m_request.verb == HTTP_HEAD) || (status == HTTP_NO_CONTENT) ||
				(static_cast<unsigned>(status) / 100 == 1))
			{
				contents.clear();
			}
			respond(session.get(), status, STD_MOVE(headers), &contents);
		} catch(HttpException &e){
			LOG_ERROR("HttpException thrown in HTTP servlet, status = ", e.status(),
				", file = ", e.file(), ", line = ", e.line());
			respond(session.get(), e.status());
			session->shutdown();
			throw;
		} catch(...){
			LOG_ERROR("Forwarding exception... shutdown the session first.");
			respond(session.get(), HTTP_SERVER_ERROR);
			session->shutdown();
			throw;
		}
	}
};

}

HttpSession::HttpSession(Move<ScopedFile> socket)
	: TcpSessionBase(STD_MOVE(socket))
	, m_state(ST_FIRST_HEADER), m_totalLength(0), m_contentLength(0), m_line()
{
}
HttpSession::~HttpSession(){
	if(m_state != ST_FIRST_HEADER){
		LOG_WARNING("Now that this HTTP session is to be destroyed, "
			"a premature request has to be discarded.");
	}
}

void HttpSession::onAllHeadersRead(){
	for(AUTO(it, m_headers.begin()); it != m_headers.end(); ++it){
		const char *const name = it->first.get();
		const std::string &val = it->second;

		if(std::strcmp(name, "Expect") == 0){
			if(val != "100-continue"){
				LOG_WARNING("Unknown HTTP header Expect: ", val);
				DEBUG_THROW(HttpException, HTTP_NOT_SUPPORTED);
			}
			respond(this, HTTP_CONTINUE);
		} else if(std::strcmp(name, "Content-Length") == 0){
			m_contentLength = boost::lexical_cast<std::size_t>(val);
			LOG_DEBUG("Content-Length: ", m_contentLength);
		}
	}
}

void HttpSession::onReadAvail(const void *data, std::size_t size){
	PROFILE_ME;

	try {
		if(!m_shutdownTimer){
			const AUTO(timeout, getHttpKeepAliveTimeout());
			LOG_DEBUG("HTTP timeout = ", timeout);

			m_shutdownTimer = TimerDaemon::registerTimer(timeout, 0, NULLPTR,
				TR1::bind(&onSessionTimeout, virtualWeakFromThis<HttpSession>(), TR1::placeholders::_1));
		}

		if(m_totalLength + size >= MAX_REQUEST_LENGTH){
			LOG_WARNING("Request size is ", m_totalLength + size,
				" and has exceeded MAX_REQUEST_LENGTH which is ", MAX_REQUEST_LENGTH);
			DEBUG_THROW(HttpException, HTTP_REQUEST_TOO_LARGE);
		}
		m_totalLength += size;

		AUTO(read, (const char *)data);
		const AUTO(end, read + size);
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
					const std::size_t questionPos = m_uri.find('?');
					if(questionPos == std::string::npos){
						m_getParams.clear();
					} else {
						m_getParams = optionalMapFromUrlEncoded(m_uri.substr(questionPos + 1));
						m_uri.erase(m_uri.begin() + questionPos, m_uri.end());
					}
					normalizeUri(m_uri);
					if((parts[2] != "HTTP/1.0") && (parts[2] != "HTTP/1.1")){
						LOG_WARNING("Unsupported HTTP version: ", parts[2]);
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
			boost::make_shared<HttpRequestJob>(virtualWeakFromThis<HttpSession>(),
				STD_MOVE(m_verb), STD_MOVE(m_uri), STD_MOVE(m_getParams),
				STD_MOVE(m_headers), STD_MOVE(m_line))->pend();

			m_state = ST_FIRST_HEADER;
			m_totalLength = 0;
			m_contentLength = 0;
			m_line.clear();

			m_uri.clear();
			m_getParams.clear();
			m_headers.clear();
		}
	} catch(HttpException &e){
		LOG_ERROR("HttpException thrown while parsing HTTP data, status = ", e.status(),
			", file = ", e.file(), ", line = ", e.line());
		respond(this, e.status());
		shutdown();
		throw;
	} catch(...){
		LOG_ERROR("Forwarding exception... shutdown the session first.");
		respond(this, HTTP_SERVER_ERROR);
		shutdown();
		throw;
	}
}
