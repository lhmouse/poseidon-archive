#include "../../precompiled.hpp"
#include "session.hpp"
#include "status.hpp"
#include "utilities.hpp"
#include "../log.hpp"
#include "../singletons/http_servlet_manager.hpp"
#include "../stream_buffer.hpp"
#include "../utilities.hpp"
#include "../exception.hpp"
#include "../job_base.hpp"
using namespace Poseidon;

namespace {

const unsigned long long MAX_REQUEST_LENGTH = 0x4000;	// 头部加正文总长度。

void respond(const boost::shared_ptr<HttpSession> &session, HttpStatus status,
	OptionalMap *headers = NULLPTR, StreamBuffer *contents = NULLPTR)
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

		headers->set("Content-Type", "text/html; charset=utf-8");
	} else {
		AUTO_REF(contentType, headers->create("Content-Type"));
		if(contentType.empty()){
			contentType.assign("text/plain; charset=utf-8");
		}
	}
	headers->set("Content-Length", boost::lexical_cast<std::string>(contents->size()));

	StreamBuffer buffer;
	buffer.put("HTTP/1.1 ");
	buffer.put(codeStatus.data(), codeStatus.size());
	buffer.put("\r\n");
	for(AUTO(it, headers->begin()); it != headers->end(); ++it){
		if(!it->second.empty()){
			buffer.put(it->first.get(), std::strlen(it->first.get()));
			buffer.put(": ");
			buffer.put(it->second.data(), it->second.size());
			buffer.put("\r\n");
		}
	}
	buffer.put("\r\n");
	buffer.splice(*contents);

	session->sendUsingMove(buffer);
}

class HttpRequestJob : public JobBase {
private:
	const boost::weak_ptr<HttpSession> m_session;

	HttpVerb m_verb;
	std::string m_uri;
	OptionalMap m_getParams;
	OptionalMap m_inHeaders;
	std::string m_inContents;

public:
	HttpRequestJob(boost::weak_ptr<HttpSession> session,
		HttpVerb verb, std::string uri, OptionalMap getParams,
		OptionalMap inHeaders, std::string inContents)
		: m_session(STD_MOVE(session))
		, m_verb(STD_MOVE(verb)), m_uri(STD_MOVE(uri)), m_getParams(STD_MOVE(getParams))
		, m_inHeaders(STD_MOVE(inHeaders)), m_inContents(STD_MOVE(inContents))
	{
	}

protected:
	void perform() const {
		const AUTO(session, m_session.lock());
		if(!session){
			LOG_WARNING("The specified HTTP session has expired.");
			return;
		}

		boost::shared_ptr<void> lockedDep;
		const AUTO(servlet, HttpServletManager::getServlet(lockedDep, m_uri));
		if(!servlet){
			LOG_WARNING("No servlet for URI ", m_uri);
			respond(session, HTTP_NOT_FOUND);
			return;
		}
		LOG_DEBUG("Dispatching http request: URI = ", m_uri, ", verb = ", stringFromHttpVerb(m_verb));
		OptionalMap headers;
		StreamBuffer contents;
		try {
			const HttpStatus status = (*servlet)(headers, contents,
				m_verb, m_getParams, m_inHeaders, m_inContents);
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

void HttpSession::onReadAvail(const void *data, std::size_t size){
	if(m_totalLength + size >= MAX_REQUEST_LENGTH){
		respond(virtualSharedFromThis<HttpSession>(), HTTP_REQUEST_TOO_LARGE);
		shutdown();
		return;
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
				m_line.resize(lineLen - 1);
			}
			if(m_state == ST_FIRST_HEADER){
				if(m_line.empty()){
					continue;
				}
				const AUTO(parts, explode<std::string>(' ', m_line, 3));
				if(parts.size() != 3){
					LOG_WARNING("Bad HTTP header: ", m_line);
					respond(virtualSharedFromThis<HttpSession>(), HTTP_BAD_REQUEST);
					shutdown();
					return;
				}
				m_verb = httpVerbFromString(parts[0].c_str());
				if(m_verb == HTTP_INVALID_VERB){
					LOG_WARNING("Bad HTTP verb: ", parts[0]);
					respond(virtualSharedFromThis<HttpSession>(), HTTP_BAD_METHOD);
					shutdown();
					return;
				}
				if(parts[1].empty() || (parts[1][0] != '/')){
					LOG_WARNING("Bad HTTP request URI: ", parts[1]);
					respond(virtualSharedFromThis<HttpSession>(), HTTP_BAD_REQUEST);
					shutdown();
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
					shutdown();
					return;
				}
				m_state = ST_HEADERS;
			} else if(!m_line.empty()){
				const std::size_t delimPos = m_line.find(':');
				if(delimPos == std::string::npos){
					LOG_WARNING("Bad HTTP header: ", m_line);
					respond(virtualSharedFromThis<HttpSession>(), HTTP_BAD_REQUEST);
					shutdown();
					return;
				}
				const char *value = m_line.c_str() + delimPos + 1;
				while(*value == ' '){
					++value;
					if(*value == 0){
						break;
					}
				}
				m_headers.add(m_line.c_str(), delimPos,
					std::string(value, m_line.c_str() + m_line.size()));
			} else {
				m_state = ST_CONTENTS;
			}
		}
		if(m_state == ST_CONTENTS){
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
		}
		m_line.clear();
	}
}
