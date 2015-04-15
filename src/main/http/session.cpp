// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include <string.h>
#include "request.hpp"
#include "exception.hpp"
#include "utilities.hpp"
#include "upgraded_session_base.hpp"
#include "../log.hpp"
#include "../singletons/http_servlet_depository.hpp"
#include "../singletons/websocket_adaptor_depository.hpp"
#include "../singletons/epoll_daemon.hpp"
#include "../stream_buffer.hpp"
#include "../string.hpp"
#include "../exception.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../hash.hpp"
#include "../websocket/session.hpp"

namespace Poseidon {

namespace Http {
	namespace {
		class RequestJob : public JobBase {
		private:
			const boost::weak_ptr<Session> m_session;
			const Request m_request;

		public:
			RequestJob(boost::weak_ptr<Session> session, Verb verb, std::string uri, unsigned version,
				OptionalMap getParams, OptionalMap headers, std::string contents)
				: m_session(STD_MOVE(session))
				, m_request(verb, STD_MOVE(uri), version, STD_MOVE(getParams), STD_MOVE(headers), STD_MOVE(contents))
			{
			}

		protected:
			boost::weak_ptr<const void> getCategory() const OVERRIDE {
				return m_session;
			}
			void perform() const OVERRIDE {
				PROFILE_ME;

				const AUTO(session, m_session.lock());
				if(!session){
					return;
				}

				try {
					const AUTO(category, session->getCategory());
					const AUTO(servlet, HttpServletDepository::get(category, m_request.uri.c_str()));
					if(!servlet){
						LOG_POSEIDON_WARNING("No handler in category ", category, " matches URI ", m_request.uri);
						DEBUG_THROW(Exception, ST_NOT_FOUND);
					}

					LOG_POSEIDON_DEBUG("Dispatching: URI = ", m_request.uri, ", verb = ", getStringFromVerb(m_request.verb));
					(*servlet)(session, m_request);

					if(m_request.version < 10001){
						session->shutdown();
					} else {
						session->setTimeout(HttpServletDepository::getKeepAliveTimeout());
					}
				} catch(TryAgainLater &){
					throw;
				} catch(Exception &e){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Exception thrown in HTTP servlet, request URI = ", m_request.uri,
						", statusCode = ", e.statusCode());
					try {
						session->sendDefault(e.statusCode(), e.headers(), false); // 不关闭连接。
					} catch(...){
						session->forceShutdown();
					}
					throw;
				} catch(...){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Forwarding exception... request URI = ", m_request.uri);
					try {
						session->sendDefault(ST_BAD_REQUEST, true); // 关闭连接。
					} catch(...){
						session->forceShutdown();
					}
					throw;
				}
			}
		};

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
			uri.erase(uri.begin() + static_cast<std::ptrdiff_t>(write), uri.end());
		}

		void onRequestTimeout(const boost::weak_ptr<Session> &weak){
			const AUTO(session, weak.lock());
			if(!session){
				return;
			}
			LOG_POSEIDON_WARNING("HTTP request times out, remote = ", session->getRemoteInfo());
			session->sendDefault(ST_REQUEST_TIMEOUT, true);
		}
		void onKeepAliveTimeout(const boost::weak_ptr<Session> &weak){
			const AUTO(tcpSession, boost::static_pointer_cast<TcpSessionBase>(weak.lock()));
			if(!tcpSession){
				return;
			}
			tcpSession->shutdown();
		}
	}

	class Session::HeaderParser {
	private:
		class ContinueResponseJob : public JobBase {
		private:
			const boost::weak_ptr<Session> m_session;

		public:
			explicit ContinueResponseJob(boost::weak_ptr<Session> session)
				: m_session(STD_MOVE(session))
			{
			}

		protected:
			boost::weak_ptr<const void> getCategory() const OVERRIDE {
				return m_session;
			}
			void perform() const OVERRIDE {
				PROFILE_ME;

				const AUTO(session, m_session.lock());
				if(!session){
					return;
				}

				session->sendDefault(ST_CONTINUE);
			}
		};

		class UpgradeToWebSocketJob : public JobBase {
		private:
			const boost::weak_ptr<Session> m_session;
			const std::string m_key;
			const boost::shared_ptr<const WebSocket::AdaptorCallback> m_adaptor;

		public:
			UpgradeToWebSocketJob(boost::weak_ptr<Session> session, std::string key,
				boost::shared_ptr<const WebSocket::AdaptorCallback> adaptor)
				: m_session(STD_MOVE(session)), m_key(STD_MOVE(key))
				, m_adaptor(STD_MOVE(adaptor))
			{
			}

		public:
			boost::weak_ptr<const void> getCategory() const OVERRIDE {
				return m_session;
			}
			void perform() const OVERRIDE {
				PROFILE_ME;

				const AUTO(session, m_session.lock());
				if(!session){
					return;
				}

				OptionalMap headers;
				headers.set("Upgrade", "websocket");
				headers.set("Connection", "Upgrade");
				headers.set("Sec-WebSocket-Accept", m_key);
				session->sendDefault(ST_SWITCHING_PROTOCOLS, STD_MOVE(headers));

				session->m_upgradedSession = boost::make_shared<WebSocket::Session>(session, m_adaptor);
			}
		};

	private:
		static void onExpect(const boost::shared_ptr<Session> &session, const std::string &val){
			if(val != "100-continue"){
				LOG_POSEIDON_WARNING("Unknown HTTP header Expect: ", val);
				DEBUG_THROW(Exception, ST_BAD_REQUEST);
			}
			enqueueJob(boost::make_shared<ContinueResponseJob>(session));
		}
		static void onContentLength(const boost::shared_ptr<Session> &session, const std::string &val){
			session->m_contentLength = boost::lexical_cast<std::size_t>(val);
			LOG_POSEIDON_DEBUG("Content-Length: ", session->m_contentLength);
		}
		static void onUpgrade(const boost::shared_ptr<Session> &session, const std::string &val){
			if(session->m_version < 10001){
				LOG_POSEIDON_WARNING("HTTP 1.1 is required to use WebSocket");
				DEBUG_THROW(Exception, ST_VERSION_NOT_SUPPORTED);
			}
			if(::strcasecmp(val.c_str(), "websocket") != 0){
				LOG_POSEIDON_WARNING("Unknown HTTP header Upgrade: ", val);
				DEBUG_THROW(Exception, ST_BAD_REQUEST);
			}
			AUTO_REF(version, session->m_headers.get("Sec-WebSocket-Version"));
			if(version != "13"){
				LOG_POSEIDON_WARNING("Unknown HTTP header Sec-WebSocket-Version: ", version);
				DEBUG_THROW(Exception, ST_BAD_REQUEST);
			}
			const AUTO(category, session->getCategory());
			AUTO(adaptor, WebSocketAdaptorDepository::get(category, session->m_uri.c_str()));
			if(!adaptor){
				LOG_POSEIDON_WARNING("No adaptor in category ", category, " matches URI ", session->m_uri);
				DEBUG_THROW(Exception, ST_NOT_FOUND);
			}

			std::string key = session->m_headers.get("Sec-WebSocket-Key");
			if(key.empty()){
				LOG_POSEIDON_WARNING("No Sec-WebSocket-Key specified.");
				DEBUG_THROW(Exception, ST_BAD_REQUEST);
			}
			key += "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
			unsigned char sha1[20];
			sha1Sum(sha1, key.data(), key.size());
			key = base64Encode(sha1, sizeof(sha1));
			enqueueJob(boost::make_shared<UpgradeToWebSocketJob>(session, STD_MOVE(key), STD_MOVE(adaptor)));
			session->m_preparedToUpgrade = true;
		}
		static void onAuthorization(const boost::shared_ptr<Session> &session, const std::string &val){
			if(!session->m_authInfo){
				return;
			}

			const std::size_t pos = val.find(' ');
			if(pos == std::string::npos){
				LOG_POSEIDON_WARNING("Bad Authorization: ", val);
				DEBUG_THROW(Exception, ST_BAD_REQUEST);
			}
			std::string temp(val);
			temp.at(pos) = 0;
			if(::strcasecmp(temp.c_str(), "basic") != 0){
				LOG_POSEIDON_WARNING("Unknown auth method: ", temp.c_str());
				DEBUG_THROW(Exception, ST_BAD_REQUEST);
			}
			temp.erase(temp.begin(), temp.begin() + static_cast<std::ptrdiff_t>(pos) + 1);
			if(!std::binary_search(session->m_authInfo->begin(), session->m_authInfo->end(), temp)){
				LOG_POSEIDON_WARNING("Invalid username or password");
				OptionalMap authHeader;
				authHeader.set("WWW-Authenticate", "Basic realm=\"Invalid username or password\"");
				DEBUG_THROW(Exception, ST_UNAUTHORIZED, STD_MOVE(authHeader));
			}
			session->m_authInfo.reset();
		}

	public:
		static void commit(const boost::shared_ptr<Session> &session){
			static const std::pair<
				const char *, void (*)(const boost::shared_ptr<Session> &, const std::string &)
				> JUMP_TABLE[] =
			{
				// 确保字母顺序。
				std::make_pair("Authorization", &onAuthorization),
				std::make_pair("Content-Length", &onContentLength),
				std::make_pair("Expect", &onExpect),
				std::make_pair("Upgrade", &onUpgrade),
			};

			for(AUTO(it, session->m_headers.begin()); it != session->m_headers.end(); ++it){
				LOG_POSEIDON_DEBUG("HTTP header: ", it->first.get(), " = ", it->second);

				AUTO(lower, BEGIN(JUMP_TABLE));
				AUTO(upper, END(JUMP_TABLE));
				for(;;){
					const AUTO(middle, lower + (upper - lower) / 2);
					const int result = std::strcmp(it->first.get(), middle->first);
					if(result == 0){
						(*middle->second)(session, it->second);
						break;
					} else if(result < 0){
						upper = middle;
					} else {
						lower = middle + 1;
					}
					if(lower == upper){
						break;
					}
				}
			}
		}
	};

	Session::Session(std::size_t category, UniqueFile socket, boost::shared_ptr<const std::vector<std::string> > authInfo)
		: TcpSessionBase(STD_MOVE(socket))
		, m_category(category)
		, m_authInfo(STD_MOVE(authInfo))
		, m_state(S_FIRST_HEADER), m_totalLength(0), m_contentLength(0)
		, m_verb(V_INVALID_VERB), m_version(10000)
		, m_preparedToUpgrade(false)
	{
		if(m_authInfo){
			bool isSorted = true;
#ifdef POSEIDON_CXX11
			isSorted = std::is_sorted(m_authInfo->begin(), m_authInfo->end());
#else
			if(m_authInfo->size() >= 2){
				for(AUTO(it, m_authInfo->begin() + 1); it != m_authInfo->end(); ++it){
					if(!(it[-1] < it[0])){
						isSorted = false;
						break;
					}
				}
			}
#endif
			if(!isSorted){
				LOG_POSEIDON_ERROR("authInfo is not sorted.");
				DEBUG_THROW(BasicException, SharedNts::observe("authInfo is not sorted"));
			}
		}
	}
	Session::~Session(){
		if(m_state != S_FIRST_HEADER){
			LOG_POSEIDON_WARNING("Now that this session is to be destroyed, a premature request has to be discarded.");
		}
	}

	void Session::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		try {
			const AUTO(maxRequestLength, HttpServletDepository::getMaxRequestLength());

			AUTO(read, static_cast<const char *>(data));
			const AUTO(end, read + size);
			for(;;){
				if((m_state == S_FIRST_HEADER) && m_upgradedSession){
					m_upgradedSession->onReadAvail(read, static_cast<std::size_t>(end - read));
					read = end;
					goto _exitFor;
				}

				while(m_state != S_CONTENTS){
					if(read == end){
						goto _exitFor;
					}
					const char ch = *read;
					++read;

					++m_totalLength;
					if(m_totalLength >= maxRequestLength){
						LOG_POSEIDON_WARNING("Request header too large: maxRequestLength = ", maxRequestLength);
						DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
					}

					if(ch != '\n'){
						m_line.push_back(ch);
						continue;
					}

					if(!m_line.empty() && (*m_line.rbegin() == '\r')){
						m_line.erase(m_line.end() - 1, m_line.end());
					}

					switch(m_state){
					case S_FIRST_HEADER:
						if(m_line.empty()){
							// m_state = S_FIRST_HEADER;
						} else {
							AUTO(parts, explode<std::string>(' ', m_line, 3));
							if(parts.size() != 3){
								LOG_POSEIDON_WARNING("Bad HTTP header: ", m_line);
								DEBUG_THROW(Exception, ST_BAD_REQUEST);
							}

							m_verb = getVerbFromString(parts[0].c_str());
							if(m_verb == V_INVALID_VERB){
								LOG_POSEIDON_WARNING("Bad HTTP verb: ", parts[0]);
								DEBUG_THROW(Exception, ST_METHOD_NOT_ALLOWED);
							}

							if(parts[1][0] != '/'){
								LOG_POSEIDON_WARNING("Bad HTTP request URI: ", parts[1]);
								DEBUG_THROW(Exception, ST_BAD_REQUEST);
							}
							parts[1] = STD_MOVE(m_uri);
							std::size_t pos = m_uri.find('#');
							if(pos != std::string::npos){
								m_uri.erase(m_uri.begin() + static_cast<std::ptrdiff_t>(pos), m_uri.end());
							}
							pos = m_uri.find('?');
							if(pos != std::string::npos){
								m_getParams = optionalMapFromUrlEncoded(m_uri.substr(pos + 1));
								m_uri.erase(m_uri.begin() + static_cast<std::ptrdiff_t>(pos), m_uri.end());
							}
							normalizeUri(m_uri);
							m_uri = STD_MOVE(urlDecode(m_uri));

							char versionMajor[16];
							char versionMinor[16];
							const int result = std::sscanf(parts[2].c_str(), "HTTP/%15[0-9].%15[0-9]%*c", versionMajor, versionMinor);
							if(result != 2){
								LOG_POSEIDON_WARNING("Bad protocol string: ", parts[2]);
								DEBUG_THROW(Exception, ST_BAD_REQUEST);
							}
							m_version = std::strtoul(versionMajor, NULLPTR, 10) * 10000 + std::strtoul(versionMinor, NULLPTR, 10);
							if((m_version != 10000) && (m_version != 10001)){
								LOG_POSEIDON_WARNING("Bad HTTP version: ", parts[2]);
								DEBUG_THROW(Exception, ST_VERSION_NOT_SUPPORTED);
							}

							m_line.clear();

							m_state = S_HEADERS;
						}
						break;

					case S_HEADERS:
						if(m_line.empty()){
							HeaderParser::commit(virtualSharedFromThis<Session>());

							m_state = S_CONTENTS;
						} else {
							std::size_t pos = m_line.find(':');
							if(pos == std::string::npos){
								LOG_POSEIDON_WARNING("Bad HTTP header: ", m_line);
								DEBUG_THROW(Exception, ST_BAD_REQUEST);
							}
							AUTO(valueBegin, m_line.begin() + static_cast<std::ptrdiff_t>(pos) + 1);
							while(*valueBegin == ' '){
								++valueBegin;
							}
							std::string value(valueBegin, m_line.end());
							m_line.erase(m_line.begin() + static_cast<std::ptrdiff_t>(pos), m_line.end());
							m_headers.append(m_line.c_str(), STD_MOVE(value));

							 m_line.clear();

							// m_state = S_HEADERS;
						}
						break;

					case S_CONTENTS:
						std::abort();

					default:
						LOG_POSEIDON_FATAL("Invalid state: ", static_cast<unsigned>(m_state));
						std::abort();
					}
				}

				if(m_authInfo){
					OptionalMap authHeader;
					authHeader.set("WWW-Authenticate", "Basic realm=\"Authentication required\"");
					DEBUG_THROW(Exception, ST_UNAUTHORIZED, STD_MOVE(authHeader));
				}

				assert(m_contentLength >= m_line.size());

				const AUTO(bytesAvail, static_cast<std::size_t>(end - read));
				const AUTO(bytesRemaining, m_contentLength - m_line.size());
				if(bytesAvail < bytesRemaining){
					m_totalLength += bytesAvail;
					if((m_totalLength < bytesAvail) || (m_totalLength > maxRequestLength)){
						LOG_POSEIDON_WARNING("Request entity too large: maxRequestLength = ", maxRequestLength);
						DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
					}
					m_line.append(read, bytesAvail);
					read = end;
					goto _exitFor;
				}
				m_totalLength += bytesRemaining;
				if((m_totalLength < bytesRemaining) || (m_totalLength > maxRequestLength)){
					LOG_POSEIDON_WARNING("Request entity too large: maxRequestLength = ", maxRequestLength);
					DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
				}
				m_line.append(read, bytesRemaining);
				read += bytesRemaining;

				if(m_upgradedSession){
					setTimeout(EpollDaemon::getTcpRequestTimeout());
					m_upgradedSession->onInitContents(m_line.data(), m_line.size());
				} else {
					onRequest(m_verb, STD_MOVE(m_uri), m_version,
						STD_MOVE(m_getParams), STD_MOVE(m_headers), STD_MOVE(m_line));
					m_verb = V_INVALID_VERB;
					m_uri.clear();
					m_version = 10000;
					m_getParams.clear();
					m_headers.clear();
				}

				m_totalLength = 0;
				m_contentLength = 0;
				m_line.clear();

				m_state = S_FIRST_HEADER;
			}
		_exitFor:
			;
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Exception thrown while parsing data, URI = ", m_uri,
				", status = ", static_cast<unsigned>(e.statusCode()));
			try {
				sendDefault(e.statusCode(), e.headers(), true);
			} catch(...){
				forceShutdown();
			}
			throw;
		} catch(...){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Forwarding exception... shutdown the session first.");
			try {
				sendDefault(ST_BAD_REQUEST, true);
			} catch(...){
				forceShutdown();
			}
			throw;
		}
	}

	void Session::onRequest(Verb verb, std::string uri, unsigned version,
		OptionalMap getParams, OptionalMap headers, std::string contents)
	{
		enqueueJob(boost::make_shared<RequestJob>(virtualWeakFromThis<Session>(),
			verb, STD_MOVE(uri), version, STD_MOVE(getParams), STD_MOVE(headers), STD_MOVE(contents)));
	}

	bool Session::send(StatusCode statusCode, OptionalMap headers, StreamBuffer contents, bool fin){
		LOG_POSEIDON_DEBUG("Making HTTP response: statusCode = ", statusCode);

		StreamBuffer data;

		char first[64];
		unsigned len = (unsigned)std::sprintf(first, "HTTP/1.1 %u ", static_cast<unsigned>(statusCode));
		data.put(first, len);
		const AUTO(desc, getStatusCodeDesc(statusCode));
		data.put(desc.descShort);
		data.put("\r\n");

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
				data.put(it->first.get());
				data.put(": ");
				data.put(it->second.data(), it->second.size());
				data.put("\r\n");
			}
		}
		data.put("\r\n");

		data.splice(contents);

		return TcpSessionBase::send(STD_MOVE(data), fin);
	}
	bool Session::sendDefault(StatusCode statusCode, OptionalMap headers, bool fin){
		LOG_POSEIDON_DEBUG("Making default HTTP response: statusCode = ", statusCode);

		StreamBuffer contents;
		if(static_cast<unsigned>(statusCode) / 100 >= 4){
			headers.set("Content-Type", "text/html; charset=utf-8");

			contents.put("<html><head><title>");
			const AUTO(desc, getStatusCodeDesc(statusCode));
			contents.put(desc.descShort);
			contents.put("</title></head><body><h1>");
			contents.put(desc.descShort);
			contents.put("</h1><hr /><p>");
			contents.put(desc.descLong);
			contents.put("</p></body></html>");
		}
		return send(statusCode, STD_MOVE(headers), STD_MOVE(contents), fin);
	}
}

}
