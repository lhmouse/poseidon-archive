#include "../precompiled.hpp"
#include "http_session.hpp"
#include "http_status.hpp"
#include "log.hpp"
#include "singletons/job_dispatcher.hpp"
#include "singletons/http_servlet_manager.hpp"
#include "stream_buffer.hpp"
#include "utilities.hpp"
#include "exception.hpp"
using namespace Poseidon;

namespace {

const unsigned long long MAX_REQUEST_LENGTH = 0x4000;	// 头部加正文总长度。

void respond(const boost::shared_ptr<HttpSession> &session, HttpStatus status,
	OptionalMap *headers = NULL, StreamBuffer *contents = NULL)
{
	LOG_DEBUG("Sending HTTP response: status = ", (unsigned)status);

	const AUTO(desc, getHttpStatusCodeDesc(status));
	const AUTO(codeStatus, boost::lexical_cast<std::string>((unsigned)status) + ' ' + desc.descShort);

	OptionalMap emptyHeaders;
	if(!headers){
		headers = &emptyHeaders;
	}

	StreamBuffer emptyContents;
	if(!contents){
		contents = &emptyContents;
	}

	if(contents->empty() && ((unsigned)status / 100 != 2)){
		contents->put("<html><head><title>");
		contents->put(codeStatus.c_str());
		contents->put("</title></head><body><h1>");
		contents->put(codeStatus.c_str());
		contents->put("</h1><hr /><p>");
		contents->put(desc.descLong);
		contents->put("</p></body></html>");

		headers->set("Content-Type", "text/html");
	} else {
		AUTO_REF(contentType, headers->create("Content-Type"));
		if(contentType.empty()){
			contentType.assign("text/plain; charset=utf-8");
		}
	}
	headers->set("Content-Length", boost::lexical_cast<std::string>(contents->size()));

	StreamBuffer buffer;
	buffer.put("HTTP/1.1 ", 9);
	buffer.put(codeStatus.data(), codeStatus.size());
	buffer.put("\r\n", 2);
	for(AUTO(it, headers->begin()); it != headers->end(); ++it){
		if(!it->second.empty()){
			buffer.put(it->first.get(), std::strlen(it->first.get()));
			buffer.put(": ", 2);
			buffer.put(it->second.data(), it->second.size());
			buffer.put("\r\n", 2);
		}
	}
	buffer.put("\r\n", 2);
	buffer.splice(*contents);

	session->sendUsingMove(buffer);
}

HttpVerb verbFromString(const std::string &verb){
	switch(verb.size()){
	case 3:
		if(verb == "GET"){
			return HTTP_GET;
		} else if(verb == "PUT"){
			return HTTP_PUT;
		}
		break;

	case 4:
		if(verb == "POST"){
			return HTTP_POST;
		} else if(verb == "HEAD"){
			return HTTP_HEAD;
		}
		break;

	case 5:
		if(verb == "TRACE"){
			return HTTP_TRACE;
		}
		break;

	case 6:
		if(verb == "DELETE"){
			return HTTP_DELETE;
		}
		break;

	case 7:
		if(verb == "CONNECT"){
			return HTTP_CONNECT;
		} else if(verb == "OPTIONS"){
			return HTTP_OPTIONS;
		}
		break;
	}
	return HTTP_INVALID_VERB;
}
const char *stringFromVerb(HttpVerb verb){
	switch(verb){
	case HTTP_GET:		return "GET";
	case HTTP_POST:		return "POST";
	case HTTP_HEAD:		return "HEAD";
	case HTTP_PUT:		return "PUT";
	case HTTP_DELETE:	return "DELETE";
	case HTTP_TRACE:	return "TRACT";
	case HTTP_CONNECT:	return "CONNECT";
	case HTTP_OPTIONS:	return "OPTIONS";
	default:			return "INVALID_VERB";
	}
}

int getHexLiteral(char ch){
	if(('0' <= ch) && (ch <= '9')){
		return ch - '0';
	} else if(('A' <= ch) && (ch <= 'F')){
		return ch - 'A' + 0x0A;
	} else if(('a' <= ch) && (ch <= 'f')){
		return ch - 'a' + 0x0A;
	}
	return -1;
}

std::string urlDecode(const std::string &source){
	std::string ret;
	const std::size_t size = source.size();
	ret.reserve(size);
	std::size_t i = 0;
	while(i < size){
		const char ch = source[i];
		++i;
		if(ch == '+'){
			ret.push_back(' ');
			continue;
		}
		if((ch != '%') || ((i + 1) >= size)){
			ret.push_back(ch);
			continue;
		}
		const int high = getHexLiteral(source[i]);
		const int low = getHexLiteral(source[i + 1]);
		if((high == -1) || (low == -1)){
			ret.push_back(ch);
			continue;
		}
		i += 2;
		ret.push_back((high << 4) | low);
	}
	return ret;
}

OptionalMap optionalMapFromUrlEncoded(const std::string &encoded){
	OptionalMap ret;
	const AUTO(parts, explode<std::string>('&', encoded));
	for(AUTO(it, parts.begin()); it != parts.end(); ++it){
		const std::size_t pos = it->find('=');
		if(pos == std::string::npos){
			ret.set(urlDecode(*it), std::string());
		} else {
			ret.set(urlDecode(it->substr(0, pos)), urlDecode(it->substr(pos + 1)));
		}
	}
	return ret;
}

class HttpRequestJob : public JobBase {
private:
	const boost::weak_ptr<HttpSession> m_session;

	const HttpVerb m_verb;
	const std::string m_uri;
	const OptionalMap m_getParams;
	const OptionalMap m_postParams;

public:
	HttpRequestJob(boost::weak_ptr<HttpSession> session, HttpVerb verb,
		std::string uri, OptionalMap getParams, OptionalMap postParams)
		: m_session(STD_MOVE(session)), m_verb(verb)
		, m_uri(uri), m_getParams(getParams), m_postParams(postParams)
	{
	}

protected:
	void perform() const {
		const AUTO(session, m_session.lock());
		if(!session){
			LOG_WARNING("The specified HTTP session has expired.");
			return;
		}

		const AUTO(servlet, HttpServletManager::getServlet(m_uri));
		if(!servlet){
			LOG_WARNING("No servlet for URI ", m_uri);
			respond(session, HTTP_NOT_FOUND);
			return;
		}
		LOG_DEBUG("Dispatching http request: URI = ", m_uri, ", verb = ", stringFromVerb(m_verb));
		OptionalMap headers;
		StreamBuffer contents;
		try {
			const HttpStatus status = (*servlet)(headers, contents, m_verb, m_getParams, m_postParams);
			respond(session, status, &headers, &contents);
		} catch(ProtocolException &e){
			LOG_ERROR("ProtocolException thrown in HTTP servlet, code = ", e.code(),
				", file = ", e.file(), ", line = ", e.line(), ", what = ", e.what());
			if(e.code() > 0){
				respond(session, (HttpStatus)e.code());
			}
		}
	}
};

}

HttpSession::HttpSession(ScopedFile &socket)
	: TcpSessionBase(socket)
	, m_headerIndex(0), m_totalLength(0), m_contentLength(0), m_line()
{
}
HttpSession::~HttpSession(){
	if(m_headerIndex != 0){
		LOG_WARNING("Now that this HTTP session is to be destroyed, "
			"a premature request has to be discarded.");
	}
}

void HttpSession::onReadAvail(const void *data, std::size_t size){
	if(m_totalLength + size >= MAX_REQUEST_LENGTH){
		respond(virtualSharedFromThis<HttpSession>(), HTTP_REQUEST_TOO_LARGE);
		shutdownRead();
		return;
	}
	m_totalLength += size;

	AUTO(read, (const char *)data);
	const AUTO(end, read + size);
	while(read != end){
		if(m_headerIndex != -1){	// -1 表示正文。
			const char ch = *read;
			++read;
			if(ch != '\n'){
				m_line.push_back(ch);
				continue;
			}
			const std::size_t lineLen = m_line.size();
			if((lineLen > 0) && (m_line[lineLen - 1] == '\r')){
				m_line.resize(lineLen - 1);
			}
			if(m_headerIndex == 0){
				if(m_line.empty()){
					continue;
				}
				const AUTO(parts, explode<std::string>(' ', m_line, 3));
				if(parts.size() != 3){
					LOG_WARNING("Bad HTTP header: ", m_line);
					respond(virtualSharedFromThis<HttpSession>(), HTTP_BAD_REQUEST);
					shutdownRead();
					return;
				}
				m_verb = verbFromString(parts[0]);
				if(m_verb == HTTP_INVALID_VERB){
					LOG_WARNING("Bad HTTP verb: ", parts[0]);
					respond(virtualSharedFromThis<HttpSession>(), HTTP_BAD_METHOD);
					shutdownRead();
					return;
				}
				if(parts[1].empty() || (parts[1][0] != '/')){
					LOG_WARNING("Bad HTTP request URI: ", parts[1]);
					respond(virtualSharedFromThis<HttpSession>(), HTTP_BAD_REQUEST);
					shutdownRead();
					return;
				}
				const std::size_t questionPos = parts[1].find('?');
				if(questionPos == std::string::npos){
					m_uri = parts[1];
					m_getParams.clear();
				} else {
					m_uri = parts[1].substr(0, questionPos);
					m_getParams = optionalMapFromUrlEncoded(parts[1].substr(questionPos + 1));
				}
				if((parts[2] != "HTTP/1.0") && (parts[2] != "HTTP/1.1")){
					LOG_WARNING("Unsupported HTTP version: ", parts[2]);
					respond(virtualSharedFromThis<HttpSession>(), HTTP_VERSION_NOT_SUP);
					shutdownRead();
					return;
				}
				++m_headerIndex;
			} else if(!m_line.empty()){
				const std::size_t delimPos = m_line.find(':');
				if(delimPos == std::string::npos){
					LOG_WARNING("Bad HTTP header: ", m_line);
					respond(virtualSharedFromThis<HttpSession>(), HTTP_BAD_REQUEST);
					shutdownRead();
					return;
				}
				const char *value = m_line.c_str() + delimPos + 1;
				while(*value == ' '){
					++value;
					if(*value == 0){
						break;
					}
				}
				const std::size_t valueLen = m_line.c_str() + m_line.size() - value;
				if(m_line.compare(0, delimPos, "Content-Length") == 0){
					m_contentLength = boost::lexical_cast<std::size_t>(std::string(value, valueLen));
				} else if(m_line.compare(0, delimPos, "Content-Type") == 0){
					m_contentType.assign(value, valueLen);
					const std::size_t semicolPos = m_contentType.find(';');
					if(semicolPos != std::string::npos){
						m_contentType.resize(semicolPos);
					}
				}
				++m_headerIndex;
			} else if(m_contentLength == 0){
				// 没有正文。
				boost::make_shared<HttpRequestJob>(
					virtualWeakFromThis<HttpSession>(),
					m_verb, m_uri, m_getParams, m_postParams
					)->pend();
				m_headerIndex = 0;
				m_totalLength = 0;
			} else {
				m_headerIndex = -1;
			}
			m_line.clear();
		} else {
			const std::size_t bytesAvail = (std::size_t)(end - read);
			const std::size_t bytesRemaining = m_contentLength - m_line.size();
			if(bytesAvail < bytesRemaining){
				m_line.append(read, bytesAvail);
				read += bytesAvail;
				continue;
			}
			m_line.append(read, bytesRemaining);
			read += bytesRemaining;

			if(m_contentType == "application/x-www-form-urlencoded"){
				m_postParams = optionalMapFromUrlEncoded(m_line);
			} else {
				LOG_WARNING("Ignored unknown content type: ", m_contentType);
			}

			boost::make_shared<HttpRequestJob>(
				virtualWeakFromThis<HttpSession>(),
				m_verb, m_uri, m_getParams, m_postParams
				)->pend();
			m_headerIndex = 0;
			m_totalLength = 0;
			m_contentLength = 0;
			m_line.clear();
		}
	}
}
