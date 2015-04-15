// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include <string.h>
#include "exception.hpp"
#include "utilities.hpp"
#include "upgraded_session_base.hpp"
#include "../log.hpp"
#include "../singletons/main_config.hpp"
#include "../singletons/epoll_daemon.hpp"
#include "../stream_buffer.hpp"
#include "../string.hpp"
#include "../exception.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace Http {
	namespace {
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

	class Session::RequestJob : public JobBase {
	private:
		const boost::weak_ptr<Session> m_session;
		const Verb m_verb;
		const std::string m_uri;
		const unsigned m_version;
		const OptionalMap m_getParams;
		const OptionalMap m_headers;
		const std::string m_contents;

	public:
		RequestJob(boost::weak_ptr<Session> session, Verb verb, std::string uri, unsigned version,
			OptionalMap getParams, OptionalMap headers, std::string contents)
			: m_session(STD_MOVE(session))
			, m_verb(verb), m_uri(STD_MOVE(uri)), m_version(version)
			, m_getParams(STD_MOVE(getParams)), m_headers(STD_MOVE(headers)), m_contents(STD_MOVE(contents))
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
				LOG_POSEIDON_DEBUG("Dispatching: URI = ", m_uri, ", verb = ", getStringFromVerb(m_verb));
				session->onRequest(m_verb, m_uri, m_version, m_getParams, m_headers, m_contents);

				if(m_version < 10001){
					session->shutdown();
				} else {
					session->setTimeout(MainConfig::getConfigFile().get<boost::uint64_t>("http_keep_alive_timeout", 0));
				}
			} catch(TryAgainLater &){
				throw;
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Exception thrown in HTTP servlet, request URI = ", m_uri,
					", statusCode = ", e.statusCode());
				try {
					session->sendDefault(e.statusCode(), e.headers(), false); // 不关闭连接。
				} catch(...){
					session->forceShutdown();
				}
				throw;
			} catch(...){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Forwarding exception... request URI = ", m_uri);
				try {
					session->sendDefault(ST_BAD_REQUEST, true); // 关闭连接。
				} catch(...){
					session->forceShutdown();
				}
				throw;
			}
		}
	};

	class Session::HeaderParser {
	private:
		static void onExpect(const boost::shared_ptr<Session> &session, const std::string &val){
			if(val != "100-continue"){
				LOG_POSEIDON_WARNING("Unknown HTTP header Expect: ", val);
				DEBUG_THROW(Exception, ST_BAD_REQUEST);
			}
			session->sendDefault(ST_CONTINUE);
		}
		static void onContentLength(const boost::shared_ptr<Session> &session, const std::string &val){
			session->m_contentLength = boost::lexical_cast<std::size_t>(val);
			LOG_POSEIDON_DEBUG("Content-Length: ", session->m_contentLength);
		}
		static void onUpgrade(const boost::shared_ptr<Session> &session, const std::string &val){
			AUTO(upgradedSession, session->onUpgrade(val,
				session->m_verb, session->m_uri, session->m_version, session->m_getParams, session->m_headers));
			if(!upgradedSession){
				LOG_POSEIDON_ERROR("Upgrade failed: ", val);
				DEBUG_THROW(Exception, ST_INTERNAL_SERVER_ERROR);
			}
			{
				const boost::mutex::scoped_lock lock(session->m_upgreadedMutex);
				session->m_upgradedSession = STD_MOVE(upgradedSession);
			}
			LOG_POSEIDON_DEBUG("Upgraded to WebSocket: remote = ", session->getRemoteInfo());
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
			temp = base64Decode(temp);
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

	Session::Session(UniqueFile socket, boost::shared_ptr<const std::vector<std::string> > authInfo)
		: TcpSessionBase(STD_MOVE(socket))
		, m_authInfo(STD_MOVE(authInfo))
		, m_state(S_FIRST_HEADER), m_totalLength(0), m_contentLength(0)
		, m_verb(V_INVALID_VERB), m_version(10000)
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
			const AUTO(maxRequestLength, MainConfig::getConfigFile().get<boost::uint64_t>("http_max_request_length", 16384));

			AUTO(read, static_cast<const char *>(data));
			const AUTO(end, read + size);
			for(;;){
				if(m_state == S_FIRST_HEADER){
					// Epoll 线程中读取 m_upgradedSession 是不需要锁的。
					if(m_upgradedSession){
						m_upgradedSession->onReadAvail(read, static_cast<std::size_t>(end - read));
						read = end;
						goto _exitFor;
					}
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
							m_uri = urlDecode(m_uri);

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

				// Epoll 线程中读取 m_upgradedSession 是不需要锁的。
				if(m_upgradedSession){
					m_upgradedSession->onInitContents(m_line.data(), m_line.size());
					setTimeout(EpollDaemon::getTcpRequestTimeout());
				} else {
					enqueueJob(boost::make_shared<RequestJob>(virtualWeakFromThis<Session>(),
						m_verb, STD_MOVE(m_uri), m_version, STD_MOVE(m_getParams), STD_MOVE(m_headers), STD_MOVE(m_line)));

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

	boost::shared_ptr<UpgradedSessionBase> Session::onUpgrade(const std::string & /* type */,
		Verb /* verb */, const std::string & /* uri */, unsigned /* version */,
		const OptionalMap & /* params */, const OptionalMap & /* headers */)
	{
		return VAL_INIT;
	}

	boost::shared_ptr<UpgradedSessionBase> Session::getUpgradedSession() const {
		const boost::mutex::scoped_lock lock(m_upgreadedMutex);
		return m_upgradedSession;
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
