// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "exception.hpp"
#include "utilities.hpp"
#include "upgraded_session_base.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../singletons/main_config.hpp"
#include "../singletons/epoll_daemon.hpp"
#include "../stream_buffer.hpp"
#include "../time.hpp"
#include "../exception.hpp"
#include "../job_base.hpp"

namespace Poseidon {

namespace Http {
	namespace {
		class SessionJobBase : public JobBase {
		private:
			const boost::weak_ptr<Session> m_session;

		protected:
			explicit SessionJobBase(const boost::shared_ptr<Session> &session)
				: m_session(session)
			{
			}

		protected:
			virtual void perform(const boost::shared_ptr<Session> &session) const = 0;

		private:
			boost::weak_ptr<const void> getCategory() const FINAL {
				return m_session;
			}
			void perform() const FINAL {
				PROFILE_ME;

				const AUTO(session, m_session.lock());
				if(!session){
					return;
				}

				try {
					perform(session);
				} catch(TryAgainLater &){
					throw;
				} catch(std::exception &e){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what());
					session->forceShutdown();
					throw;
				} catch(...){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown.");
					session->forceShutdown();
					throw;
				}
			}
		};
	}

	class Session::ContinueJob : public SessionJobBase {
	public:
		explicit ContinueJob(const boost::shared_ptr<Session> &session)
			: SessionJobBase(session)
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->sendDefault(ST_CONTINUE);
		}
	};

	class Session::RequestJob : public SessionJobBase {
	private:
		const RequestHeaders m_requestHeaders;
		const StreamBuffer m_entity;

	public:
		RequestJob(const boost::shared_ptr<Session> &session, RequestHeaders requestHeaders, StreamBuffer entity)
			: SessionJobBase(session)
			, m_requestHeaders(STD_MOVE(requestHeaders)), m_entity(STD_MOVE(entity))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			try {
				LOG_POSEIDON_DEBUG("Dispatching request: URI = ", m_requestHeaders.uri);

				session->onRequest(m_requestHeaders, m_entity);

				const AUTO_REF(keepAliveStr, m_requestHeaders.headers.get("Connection"));
				if((m_requestHeaders.version < 10001)
					? (::strcasecmp(keepAliveStr.c_str(), "Keep-Alive") == 0)	// HTTP 1.0
					: (::strcasecmp(keepAliveStr.c_str(), "Close") != 0))		// HTTP 1.1
				{
					session->setTimeout(MainConfig::getConfigFile().get<boost::uint64_t>("http_keep_alive_timeout", 5000));
				} else {
					session->shutdown();
				}
			} catch(TryAgainLater &){
				throw;
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Exception thrown in HTTP servlet: URI = ", m_requestHeaders.uri,
					", statusCode = ", e.statusCode());
				try {
					session->sendDefault(e.statusCode(), e.headers(), false); // 不关闭连接。
				} catch(...){
					session->forceShutdown();
				}
			}
		}
	};

	class Session::ErrorJob : public SessionJobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;

		const StatusCode m_statusCode;
		const OptionalMap m_headers;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session, StatusCode statusCode, OptionalMap headers)
			: SessionJobBase(session)
			, m_guard(session)
			, m_statusCode(statusCode), m_headers(STD_MOVE(headers))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->sendDefault(m_statusCode, m_headers, true);
		}
	};

	Session::Session(UniqueFile socket)
		: TcpSessionBase(STD_MOVE(socket))
		, m_sizeTotal(0), m_expectingNewLine(true), m_sizeExpecting(0), m_state(S_FIRST_HEADER)
		, m_requestHeaders()
	{
	}
	Session::~Session(){
		if((m_state != S_FIRST_HEADER) && (m_state != S_UPGRADED)){
			LOG_POSEIDON_WARNING("Now that this session is to be destroyed, a premature request has to be discarded.");
		}
	}

	void Session::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		if(m_state == S_UPGRADED){
			const AUTO(upgradedSession, getUpgradedSession());
			if(upgradedSession){
				upgradedSession->onReadAvail(data, size);
				return;
			}
			LOG_POSEIDON_WARNING("Session has not been fully upgraded. Abort.");
			DEBUG_THROW(BasicException, SSLIT("Session has not been fully upgraded"));
		}

		try {
			const AUTO(maxRequestLength, MainConfig::getConfigFile().get<boost::uint64_t>("http_max_request_length", 16384));

			m_received.put(data, size);

			for(;;){
				if(m_state == S_UPGRADED){
					if(!m_received.empty()){
						LOG_POSEIDON_WARNING("Junk data received after upgrading.");
						DEBUG_THROW(BasicException, SSLIT("Junk data received after upgrading"));
					}
					break;
				}

				boost::uint64_t sizeTotal;
				bool gotExpected;
				if(m_expectingNewLine){
					struct Helper {
						static bool traverseCallback(void *ctx, const void *data, std::size_t size){
							AUTO_REF(lfOffset, *static_cast<std::size_t *>(ctx));

							const AUTO(pos, std::memchr(data, '\n', size));
							if(!pos){
								lfOffset += size;
								return true;
							}
							lfOffset += static_cast<std::size_t>(static_cast<const char *>(pos) - static_cast<const char *>(data));
							return false;
						}
					};

					std::size_t lfOffset = 0;
					if(m_received.traverse(&Helper::traverseCallback, &lfOffset)){
						// 没找到换行符。
						sizeTotal = m_sizeTotal + m_received.size();
						gotExpected = false;
					} else {
						// 找到了。
						m_sizeExpecting = lfOffset + 1;
						sizeTotal = m_sizeTotal + m_sizeExpecting;
						gotExpected = true;
					}
				} else {
					if(m_received.size() < m_sizeExpecting){
						if(m_sizeExpecting > maxRequestLength){
							LOG_POSEIDON_WARNING("Request too large: sizeExpecting = ", m_sizeExpecting);
							DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
						}
						sizeTotal = m_sizeTotal + m_received.size();
						gotExpected = false;
					} else {
						sizeTotal = m_sizeTotal + m_sizeExpecting;
						gotExpected = true;
					}
				}
				if(sizeTotal > maxRequestLength){
					LOG_POSEIDON_WARNING("Request too large: maxRequestLength = ", maxRequestLength);
					DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
				}
				if(!gotExpected){
					break;
				}
				m_sizeTotal = sizeTotal;

				AUTO(expected, m_received.cut(m_sizeExpecting));
				if(m_expectingNewLine){
					expected.unput(); // '\n'
					if(expected.back() == '\r'){
						expected.unput();
					}
				}

				switch(m_state){
				case S_FIRST_HEADER:
					if(!expected.empty()){
						m_requestHeaders = RequestHeaders();
						m_chunkedEntity.clear();

						std::string line;
						expected.dump(line);

						std::size_t pos = line.find(' ');
						if(pos == std::string::npos){
							LOG_POSEIDON_WARNING("Bad request header: expecting verb, line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						line[pos] = 0;
						m_requestHeaders.verb = getVerbFromString(line.c_str());
						if(m_requestHeaders.verb == V_INVALID_VERB){
							LOG_POSEIDON_WARNING("Bad verb: ", line.c_str());
							DEBUG_THROW(Exception, ST_NOT_IMPLEMENTED);
						}
						line.erase(0, pos + 1);

						pos = line.find(' ');
						if(pos == std::string::npos){
							LOG_POSEIDON_WARNING("Bad request header: expecting URI end, line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						m_requestHeaders.uri.assign(line, 0, pos);
						line.erase(0, pos + 1);

						long verEnd = 0;
						char verMajorStr[16], verMinorStr[16];
						if(std::sscanf(line.c_str(), "HTTP/%15[0-9].%15[0-9]%ln", verMajorStr, verMinorStr, &verEnd) != 2){
							LOG_POSEIDON_WARNING("Bad request header: expecting HTTP version, line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						if(static_cast<unsigned long>(verEnd) != line.size()){
							LOG_POSEIDON_WARNING("Bad request header: junk after HTTP version, line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						m_requestHeaders.version = std::strtoul(verMajorStr, NULLPTR, 10) * 10000 + std::strtoul(verMinorStr, NULLPTR, 10);
						if((m_requestHeaders.version != 10000) && (m_requestHeaders.version != 10001)){
							LOG_POSEIDON_WARNING("Bad request header: HTTP version not supported, verMajorStr = ", verMajorStr,
								", verMinorStr = ", verMinorStr);
							DEBUG_THROW(Exception, ST_VERSION_NOT_SUPPORTED);
						}

						pos = m_requestHeaders.uri.find('?');
						if(pos != std::string::npos){
							m_requestHeaders.getParams = optionalMapFromUrlEncoded(m_requestHeaders.uri.substr(pos + 1));
							m_requestHeaders.uri.erase(pos);
						}

						m_expectingNewLine = true;
						m_state = S_HEADERS;
					} else {
						// m_state = S_FIRST_HEADER;
					}
					break;

				case S_HEADERS:
					if(!expected.empty()){
						std::string line;
						expected.dump(line);

						std::size_t pos = line.find(':');
						if(pos == std::string::npos){
							LOG_POSEIDON_WARNING("Invalid HTTP header: line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						m_requestHeaders.headers.append(SharedNts(line.c_str(), pos),
							line.substr(line.find_first_not_of(' ', pos + 1)));

						m_expectingNewLine = true;
						// m_state = S_HEADERS;
					} else {
						boost::uint64_t sizeExpecting;

						const AUTO_REF(transferEncodingStr, m_requestHeaders.headers.get("Transfer-Encoding"));
						if(transferEncodingStr.empty() || (::strcasecmp(transferEncodingStr.c_str(), "identity") == 0)){
							const AUTO_REF(contentLengthStr, m_requestHeaders.headers.get("Content-Length"));
							if(contentLengthStr.empty()){
								sizeExpecting = 0;
							} else {
								char *endptr;
								sizeExpecting = ::strtoull(contentLengthStr.c_str(), &endptr, 10);
								if(*endptr){
									LOG_POSEIDON_WARNING("Bad request header Content-Length: ", contentLengthStr);
									DEBUG_THROW(Exception, ST_BAD_REQUEST);
								}
								if(sizeExpecting == CONTENT_CHUNKED){
									LOG_POSEIDON_WARNING("Inacceptable Content-Length: ", contentLengthStr);
									DEBUG_THROW(Exception, ST_BAD_REQUEST);
								}
							}
						} else if(::strcasecmp(transferEncodingStr.c_str(), "chunked") == 0){
							sizeExpecting = CONTENT_CHUNKED;
						} else {
							LOG_POSEIDON_WARNING("Unsupported Transfer-Encoding: ", transferEncodingStr);
							DEBUG_THROW(Exception, ST_NOT_ACCEPTABLE);
						}

						AUTO(upgradedSession, onRequestHeaders(m_requestHeaders, sizeExpecting));
						if(upgradedSession){
							{
								const boost::mutex::scoped_lock lock(m_upgradedSessionMutex);
								m_upgradedSession = STD_MOVE(upgradedSession);
							}

							// m_expectingNewLine = true;
							m_state = S_UPGRADED;
						} else if(sizeExpecting == CONTENT_CHUNKED){
							m_expectingNewLine = true;
							m_state = S_CHUNK_HEADER;
						} else {
							m_expectingNewLine = false;
							m_sizeExpecting = sizeExpecting;
							m_state = S_IDENTITY;
						}
					}
					break;

				case S_UPGRADED:
					std::abort();
					break;

				case S_IDENTITY:
					enqueueJob(boost::make_shared<RequestJob>(
						virtualSharedFromThis<Session>(), STD_MOVE(m_requestHeaders), STD_MOVE(expected)));

					m_sizeTotal = 0;
					m_expectingNewLine = true;
					m_state = S_FIRST_HEADER;
					break;

				case S_CHUNK_HEADER:
					if(!expected.empty()){
						std::string line;
						expected.dump(line);

						char *endptr;
						const boost::uint64_t chunkSize = ::strtoull(line.c_str(), &endptr, 16);
						if(*endptr && (*endptr != ' ')){
							LOG_POSEIDON_WARNING("Bad chunk header: ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						if(chunkSize == 0){
							m_expectingNewLine = true;
							m_state = S_CHUNKED_TRAILER;
						} else {
							m_expectingNewLine = false;
							m_sizeExpecting = chunkSize;
							m_state = S_CHUNK_DATA;
						}
					} else {
						// chunk-data 后面应该有一对 CRLF。我们在这里处理这种情况。
					}
					break;

				case S_CHUNK_DATA:
					m_chunkedEntity.splice(expected);

					m_expectingNewLine = true;
					m_state = S_CHUNK_HEADER;
					break;

				case S_CHUNKED_TRAILER:
					if(!expected.empty()){
						std::string line;
						expected.dump(line);

						std::size_t pos = line.find(':');
						if(pos == std::string::npos){
							LOG_POSEIDON_WARNING("Invalid HTTP header: line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						m_requestHeaders.headers.append(SharedNts(line.c_str(), pos), line.substr(line.find_first_not_of(' ', pos + 1)));

						m_expectingNewLine = true;
						// m_state = S_CHUNKED_TRAILER;
					} else {
						enqueueJob(boost::make_shared<RequestJob>(
							virtualSharedFromThis<Session>(), STD_MOVE(m_requestHeaders), STD_MOVE(m_chunkedEntity)));

						m_sizeTotal = 0;
						m_expectingNewLine = true;
						m_state = S_FIRST_HEADER;
					}
					break;

				default:
					LOG_POSEIDON_ERROR("Unknown state: ", static_cast<unsigned>(m_state));
					std::abort();
				}
			}
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Http::Exception thrown while parsing data, URI = ", m_requestHeaders.uri,
				", status = ", static_cast<unsigned>(e.statusCode()));
			try {
				enqueueJob(boost::make_shared<ErrorJob>(
					virtualSharedFromThis<Session>(), e.statusCode(), e.headers()));
				shutdown();
			} catch(...){
				forceShutdown();
			}
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown while parsing data, URI = ", m_requestHeaders.uri,
				", what = ", e.what());
			try {
				enqueueJob(boost::make_shared<ErrorJob>(
					virtualSharedFromThis<Session>(), static_cast<StatusCode>(ST_BAD_REQUEST), OptionalMap()));
				shutdown();
			} catch(...){
				forceShutdown();
			}
		}
	}
	void Session::onReadHup() NOEXCEPT {
		PROFILE_ME;

		if(m_state == S_UPGRADED){
			const AUTO(upgradedSession, getUpgradedSession());
			if(upgradedSession){
				upgradedSession->onReadHup();
			}
		}
	}

	boost::shared_ptr<UpgradedSessionBase> Session::onRequestHeaders(
		const RequestHeaders &requestHeaders, boost::uint64_t contentLength)
	{
		(void)contentLength;

		const AUTO_REF(expectStr, requestHeaders.headers.get("Expect"));
		if(!expectStr.empty()){
			if(::strcasecmp(expectStr.c_str(), "100-continue") == 0){
				enqueueJob(boost::make_shared<ContinueJob>(virtualSharedFromThis<Session>()));
			} else {
				LOG_POSEIDON_WARNING("Unknown HTTP header Expect: ", expectStr);
				DEBUG_THROW(Exception, ST_BAD_REQUEST);
			}
		}

		return VAL_INIT;
	}

	boost::shared_ptr<UpgradedSessionBase> Session::getUpgradedSession() const {
		const boost::mutex::scoped_lock lock(m_upgradedSessionMutex);
		return m_upgradedSession;
	}

	bool Session::send(ResponseHeaders responseHeaders, StreamBuffer entity, bool fin){
		PROFILE_ME;
		LOG_POSEIDON_DEBUG("Making HTTP response: statusCode = ", responseHeaders.statusCode);

		StreamBuffer data;

		const unsigned verMajor = responseHeaders.version / 10000, verMinor = responseHeaders.version % 10000;
		const unsigned statusCode = static_cast<unsigned>(responseHeaders.statusCode);
		char temp[64];
		unsigned len = (unsigned)std::sprintf(temp, "HTTP/%u.%u %u ", verMajor,verMinor, statusCode);
		data.put(temp, len);
		data.put(responseHeaders.reason);
		data.put("\r\n");

		if(!entity.empty() && !responseHeaders.headers.has("Content-Type")){
			responseHeaders.headers.set("Content-Type", "text/plain; charset=utf-8");
		}
		responseHeaders.headers.set("Content-Length", boost::lexical_cast<std::string>(entity.size()));
		for(AUTO(it, responseHeaders.headers.begin()); it != responseHeaders.headers.end(); ++it){
			data.put(it->first.get());
			data.put(": ");
			data.put(it->second.data(), it->second.size());
			data.put("\r\n");
		}
		data.put("\r\n");

		data.splice(entity);
		return TcpSessionBase::send(STD_MOVE(data), fin);
	}

	bool Session::send(StatusCode statusCode, OptionalMap headers, StreamBuffer entity, bool fin){
		PROFILE_ME;
		LOG_POSEIDON_DEBUG("Making HTTP/1.1 response: statusCode = ", statusCode);

		ResponseHeaders responseHeaders;
		responseHeaders.version = 10001;
		responseHeaders.statusCode = statusCode;
		responseHeaders.reason = getStatusCodeDesc(statusCode).descShort;
		responseHeaders.headers = STD_MOVE(headers);
		return send(STD_MOVE(responseHeaders), STD_MOVE(entity), fin);
	}

	bool Session::sendDefault(StatusCode statusCode, OptionalMap headers, bool fin){
		PROFILE_ME;
		LOG_POSEIDON_DEBUG("Making default HTTP/1.1 response: statusCode = ", statusCode, ", fin = ", fin);

		StreamBuffer entity;
		if(static_cast<unsigned>(statusCode) / 100 >= 4){
			headers.set("Content-Type", "text/html; charset=utf-8");

			entity.put("<html><head><title>");
			const AUTO(desc, getStatusCodeDesc(statusCode));
			entity.put(desc.descShort);
			entity.put("</title></head><body><h1>");
			entity.put(desc.descShort);
			entity.put("</h1><hr /><p>");
			entity.put(desc.descLong);
			entity.put("</p></body></html>");
		}
		return send(statusCode, STD_MOVE(headers), STD_MOVE(entity), fin);
	}
}

}
