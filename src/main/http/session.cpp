// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include <boost/bind.hpp>
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

StreamBuffer makeResponse(HttpStatus status, OptionalMap headers, StreamBuffer contents){
	LOG_POSEIDON_DEBUG("Making HTTP response: status = ", static_cast<unsigned>(status));

	StreamBuffer ret;

	char first[64];
	const unsigned firstLen = std::sprintf(first, "HTTP/1.1 %u ", static_cast<unsigned>(status));
	ret.put(first, firstLen);
	const AUTO(desc, getHttpStatusDesc(status));
	ret.put(desc.descShort);
	ret.put("\r\n");

	if(!contents.empty()){
		AUTO_REF(contentType, headers.create("Content-Type")->second);
		if(contentType.empty()){
			contentType.assign("text/plain; charset=utf-8");
		}
	}
	AUTO_REF(contentLength, headers.create("Content-Length")->second);
	boost::lexical_cast<std::string>(contents.size()).swap(contentLength);

	for(AUTO(it, headers.begin()); it != headers.end(); ++it){
		if(!it->second.empty()){
			ret.put(it->first.get());
			ret.put(": ");
			ret.put(it->second.data(), it->second.size());
			ret.put("\r\n");
		}
	}
	ret.put("\r\n");

	ret.splice(contents);

	return ret;
}
StreamBuffer makeDefaultResponse(HttpStatus status, OptionalMap headers){
	LOG_POSEIDON_DEBUG("Making default HTTP response: status = ", static_cast<unsigned>(status));

	StreamBuffer contents;
	if(static_cast<unsigned>(status) / 100 >= 4){
		headers.set("Content-Type", "text/html; charset=utf-8");

		contents.put("<html><head><title>");
		const AUTO(desc, getHttpStatusDesc(status));
		contents.put(desc.descShort);
		contents.put("</title></head><body><h1>");
		contents.put(desc.descShort);
		contents.put("</h1><hr /><p>");
		contents.put(desc.descLong);
		contents.put("</p></body></html>");
	}
	return makeResponse(status, STD_MOVE(headers), STD_MOVE(contents));
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

void onRequestTimeout(const boost::weak_ptr<HttpSession> &observer){
	const AUTO(session, observer.lock());
	if(session){
		LOG_POSEIDON_WARN("HTTP request times out, remote = ", session->getRemoteInfo());
		session->sendDefault(HTTP_REQUEST_TIMEOUT, true);
	}
}
void onKeepAliveTimeout(const boost::weak_ptr<HttpSession> &observer){
	const AUTO(session, observer.lock());
	if(session){
		static_cast<TcpSessionBase *>(session.get())->send(StreamBuffer(), true);
	}
}

class HttpRequestJob : public JobBase {
private:
	const boost::shared_ptr<HttpSession> m_session;

	const HttpVerb m_verb;
	const std::string m_uri;
	const unsigned m_version;

	OptionalMap m_getParams;
	OptionalMap m_headers;
	std::string m_contents;

public:
	HttpRequestJob(boost::shared_ptr<HttpSession> session,
		HttpVerb verb, std::string uri, unsigned version,
		OptionalMap getParams, OptionalMap headers,
		std::string contents)
		: m_session(STD_MOVE(session))
		, m_verb(verb), m_uri(STD_MOVE(uri)), m_version(version)
		, m_getParams(STD_MOVE(getParams)), m_headers(STD_MOVE(headers))
		, m_contents(STD_MOVE(contents))
	{
	}

protected:
	void perform(){
		PROFILE_ME;
		assert(!m_uri.empty());

		try {
			const AUTO(servlet, HttpServletManager::getServlet(m_session->getCategory(), m_uri.c_str()));
			if(!servlet){
				LOG_POSEIDON_WARN("No handler matches URI ", m_uri);
				DEBUG_THROW(HttpException, HTTP_NOT_FOUND);
			}

			HttpRequest request;
			request.verb = m_verb;
			request.uri = m_uri;
			request.getParams.swap(m_getParams);
			request.headers.swap(m_headers);
			request.contents.swap(m_contents);

			LOG_POSEIDON_DEBUG("Dispatching: URI = ", m_uri, ", verb = ", stringFromHttpVerb(m_verb));
			(*servlet)(m_session, STD_MOVE(request));
		} catch(HttpException &e){
			LOG_POSEIDON_ERROR("HttpException thrown in HTTP servlet, request URI = ", m_uri,
				", status = ", static_cast<unsigned>(e.status()));
			m_session->sendDefault(e.status(), e.headers(), true);
			throw;
		} catch(...){
			LOG_POSEIDON_ERROR("Forwarding exception... request URI = ", m_uri);
			m_session->sendDefault(HTTP_BAD_REQUEST, true);
			throw;
		}
	}
};

}

HttpSession::HttpSession(std::size_t category, ScopedFile socket)
	: TcpSessionBase(STD_MOVE(socket))
	, m_category(category)
	, m_state(ST_FIRST_HEADER), m_totalLength(0), m_contentLength(0)
	, m_verb(HTTP_GET), m_version(10000)
{
}
HttpSession::~HttpSession(){
	if(m_state != ST_FIRST_HEADER){
		LOG_POSEIDON_WARN(
			"Now that this session is to be destroyed, a premature request has to be discarded.");
	}
}

void HttpSession::onReadAvail(const void *data, std::size_t size){
	PROFILE_ME;

	if((m_state == ST_FIRST_HEADER) && m_upgradedSession){
		m_upgradedSession->onReadAvail(data, size);
		return;
	}

	AUTO(read, (const char *)data);
	const AUTO(end, read + size);
	try {
		const std::size_t maxRequestLength = HttpServletManager::getMaxRequestLength();
		if(m_totalLength + size >= maxRequestLength){
			LOG_POSEIDON_WARN("Request size is ", m_totalLength + size, ", max = ", maxRequestLength);
			DEBUG_THROW(HttpException, HTTP_REQUEST_ENTITY_TOO_LARGE);
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
						LOG_POSEIDON_WARN("Bad HTTP header: ", m_line);
						DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
					}

					m_verb = httpVerbFromString(parts[0].c_str());
					if(m_verb == HTTP_INVALID_VERB){
						LOG_POSEIDON_WARN("Bad HTTP verb: ", parts[0]);
						DEBUG_THROW(HttpException, HTTP_METHOD_NOT_ALLOWED);
					}

					if(parts[1].empty() || (parts[1][0] != '/')){
						LOG_POSEIDON_WARN("Bad HTTP request URI: ", parts[1]);
						DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
					}
					parts[1].swap(m_uri);
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
					urlDecode(m_uri).swap(m_uri);

					char versionMajor[16];
					char versionMinor[16];
					const int result = std::sscanf(parts[2].c_str(),
						"HTTP/%15[0-9].%15[0-9]%*c", versionMajor, versionMinor);
					if(result != 2){
						LOG_POSEIDON_WARN("Bad protocol string: ", parts[2]);
						DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
					}
					m_version = std::atoi(versionMajor) * 10000 + std::atoi(versionMinor);
					if((m_version != 10000) && (m_version != 10001)){
						LOG_POSEIDON_WARN("Bad HTTP version: ", parts[2]);
						DEBUG_THROW(HttpException, HTTP_VERSION_NOT_SUPPORTED);
					}

					m_state = ST_HEADERS;
				} else if(!m_line.empty()){
					const std::size_t delimPos = m_line.find(':');
					if(delimPos == std::string::npos){
						LOG_POSEIDON_WARN("Bad HTTP header: ", m_line);
						DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
					}
					AUTO(valueBegin, m_line.begin() + delimPos + 1);
					while(*valueBegin == ' '){
						++valueBegin;
					}
					std::string value(valueBegin, m_line.end());
					m_line.erase(m_line.begin() + delimPos, m_line.end());
					m_headers.append(m_line.c_str(), STD_MOVE(value));
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

			if(m_authInfo){
				OptionalMap authHeader;
				authHeader.set("WWW-Authenticate", "Basic realm=\"Authentication required\"");
				DEBUG_THROW(HttpException, HTTP_UNAUTHORIZED, STD_MOVE(authHeader));
			}

			if(m_upgradedSession){
				m_shutdownTimer.reset();

				m_upgradedSession->onInitContents(m_line.data(), m_line.size());

				m_totalLength = 0;
				m_contentLength = 0;
				m_line.clear();

				if(read != end){
					m_upgradedSession->onReadAvail(read, end - read);
				}
				read = end;
				break;
			}

			m_shutdownTimer = TimerDaemon::registerTimer(HttpServletManager::getKeepAliveTimeout(), 0,
				boost::bind(&onKeepAliveTimeout, virtualWeakFromThis<HttpSession>()));

			boost::make_shared<HttpRequestJob>(virtualSharedFromThis<HttpSession>(),
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
	} catch(HttpException &e){
		LOG_POSEIDON_ERROR("HttpException thrown while parsing data, URI = ", m_uri,
			", status = ", static_cast<unsigned>(e.status()));
		sendDefault(e.status(), e.headers(), true);
		throw;
	} catch(...){
		LOG_POSEIDON_ERROR("Forwarding exception... shutdown the session first.");
		sendDefault(HTTP_BAD_REQUEST, true);
		throw;
	}
}

void HttpSession::setRequestTimeout(unsigned long long timeout){
	LOG_POSEIDON_DEBUG("Setting request timeout to ", timeout);

	if(timeout == 0){
		m_shutdownTimer.reset();
	} else {
		m_shutdownTimer = TimerDaemon::registerTimer(timeout, 0,
			boost::bind(&onRequestTimeout, virtualWeakFromThis<HttpSession>()));
	}
}

void HttpSession::onAllHeadersRead(){
	struct Dispatcher {
		static void onExpect(HttpSession *session, const std::string &val){
			if(val != "100-continue"){
				LOG_POSEIDON_WARN("Unknown HTTP header Expect: ", val);
				DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
			}
			session->sendDefault(HTTP_CONTINUE);
		}
		static void onContentLength(HttpSession *session, const std::string &val){
			session->m_contentLength = boost::lexical_cast<std::size_t>(val);
			LOG_POSEIDON_DEBUG("Content-Length: ", session->m_contentLength);
		}
		static void onUpgrade(HttpSession *session, const std::string &val){
			if(session->m_version < 10001){
				LOG_POSEIDON_WARN("HTTP 1.1 is required to use WebSocket");
				DEBUG_THROW(HttpException, HTTP_VERSION_NOT_SUPPORTED);
			}
			if(::strcasecmp(val.c_str(), "websocket") != 0){
				LOG_POSEIDON_WARN("Unknown HTTP header Upgrade: ", val);
				DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
			}
			AUTO_REF(version, session->m_headers.get("Sec-WebSocket-Version"));
			if(version != "13"){
				LOG_POSEIDON_WARN("Unknown HTTP header Sec-WebSocket-Version: ", version);
				DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
			}

			std::string key = session->m_headers.get("Sec-WebSocket-Key");
			if(key.empty()){
				LOG_POSEIDON_WARN("No Sec-WebSocket-Key specified.");
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
			session->sendDefault(HTTP_SWITCHING_PROTOCOLS, STD_MOVE(headers));

			session->m_upgradedSession = boost::make_shared<WebSocketSession>(
				session->virtualSharedFromThis<HttpSession>());
			LOG_POSEIDON_INFO("Upgraded to WebSocketSession, remote = ", session->getRemoteInfo());
		}
		static void onAuthorization(HttpSession *session, const std::string &val){
			if(!session->m_authInfo){
				return;
			}

			const std::size_t pos = val.find(' ');
			if(pos == std::string::npos){
				LOG_POSEIDON_WARN("Bad Authorization: ", val);
				DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
			}
			std::string temp(val);
			temp.at(pos) = 0;
			if(::strcasecmp(temp.c_str(), "basic") != 0){
				LOG_POSEIDON_WARN("Unknown auth method: ", temp.c_str());
				DEBUG_THROW(HttpException, HTTP_BAD_REQUEST);
			}
			temp.erase(temp.begin(), temp.begin() + pos + 1);
			if(session->m_authInfo->find(temp) == session->m_authInfo->end()){
				LOG_POSEIDON_WARN("Invalid username or password");
				OptionalMap authHeader;
				authHeader.set("WWW-Authenticate", "Basic realm=\"Invalid username or password\"");
				DEBUG_THROW(HttpException, HTTP_UNAUTHORIZED, STD_MOVE(authHeader));
			}
			session->m_authInfo.reset();
		}
	};

	static const std::pair<
		const char *, void (*)(HttpSession *, const std::string &)
		> JUMP_TABLE[] =
	{
		// 确保字母顺序。
		std::make_pair("Authorization", &Dispatcher::onAuthorization),
		std::make_pair("Content-Length", &Dispatcher::onContentLength),
		std::make_pair("Expect", &Dispatcher::onExpect),
		std::make_pair("Upgrade", &Dispatcher::onUpgrade),
	};

	for(AUTO(it, m_headers.begin()); it != m_headers.end(); ++it){
		LOG_POSEIDON_DEBUG("HTTP header: ", it->first.get(), " = ", it->second);

		AUTO(lower, BEGIN(JUMP_TABLE));
		AUTO(upper, END(JUMP_TABLE));
		VALUE_TYPE(JUMP_TABLE[0].second) found = VAL_INIT;
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
			(*found)(this, it->second);
		}
	}
}

bool HttpSession::send(HttpStatus status, OptionalMap headers, StreamBuffer contents, bool final){
	return TcpSessionBase::send(makeResponse(status, STD_MOVE(headers), STD_MOVE(contents)), final);
}
bool HttpSession::sendDefault(HttpStatus status, OptionalMap headers, bool final){
	return TcpSessionBase::send(makeDefaultResponse(status, STD_MOVE(headers)), final);
}
