// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_session.hpp"
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "exception.hpp"
#include "utilities.hpp"
#include "upgraded_low_level_session_base.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../singletons/main_config.hpp"
#include "../singletons/epoll_daemon.hpp"
#include "../stream_buffer.hpp"
#include "../time.hpp"
#include "../string.hpp"

namespace Poseidon {

namespace Http {
	namespace {
		const std::string IDENTITY_STRING	= "identity";
		const std::string CHUNKED_STRING	= "chunked";
	}

	LowLevelSession::LowLevelSession(UniqueFile socket)
		: TcpSessionBase(STD_MOVE(socket))
		, m_sizeTotal(0), m_expectingNewLine(true), m_sizeExpecting(0), m_state(S_FIRST_HEADER)
		, m_requestHeaders()
	{
	}
	LowLevelSession::~LowLevelSession(){
		if(m_state != S_FIRST_HEADER){
			LOG_POSEIDON_WARNING("Now that this session is to be destroyed, a premature request has to be discarded.");
		}
	}

	void LowLevelSession::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		// epoll 线程读取 m_upgradedSession 不需要锁。
		if((m_state == S_FIRST_HEADER) && m_upgradedSession){
			m_upgradedSession->onReadAvail(data, size);
			return;
		}

		try {
			const AUTO(maxRequestLength, MainConfig::getConfigFile().get<boost::uint64_t>("http_max_request_length", 16384));

			m_received.put(data, size);

			for(;;){
				// epoll 线程读取 m_upgradedSession 不需要锁。
				if((m_state == S_FIRST_HEADER) && m_upgradedSession){
					const AUTO(size, m_received.size());
					if(size > 0){
						boost::scoped_array<char> data(new char[size]);
						m_received.get(data.get(), size);
						m_upgradedSession->onReadAvail(data.get(), size);
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
						AUTO(transferEncoding, explode<std::string>(',', m_requestHeaders.headers.get("Transfer-Encoding")));
						for(AUTO(it, transferEncoding.begin()); it != transferEncoding.end(); ++it){
							*it = toLowerCase(trim(STD_MOVE(*it)));
						}
						std::sort(transferEncoding.begin(), transferEncoding.end());
						AUTO(range, std::equal_range(transferEncoding.begin(), transferEncoding.end(), IDENTITY_STRING));
						transferEncoding.erase(range.first, range.second);

						boost::uint64_t contentLength;
						if(!transferEncoding.empty()){
							range = std::equal_range(transferEncoding.begin(), transferEncoding.end(), CHUNKED_STRING);
							transferEncoding.erase(range.first, range.second);
							contentLength = CONTENT_CHUNKED;
						} else {
							const AUTO_REF(contentLengthStr, m_requestHeaders.headers.get("Content-Length"));
							if(contentLengthStr.empty()){
								contentLength = 0;
							} else {
								char *endptr;
								contentLength = ::strtoull(contentLengthStr.c_str(), &endptr, 10);
								if(*endptr){
									LOG_POSEIDON_WARNING("Bad request header Content-Length: ", contentLengthStr);
									DEBUG_THROW(Exception, ST_BAD_REQUEST);
								}
								if(contentLength > CONTENT_LENGTH_MAX){
									LOG_POSEIDON_WARNING("Inacceptable Content-Length: ", contentLengthStr);
									DEBUG_THROW(Exception, ST_BAD_REQUEST);
								}
							}
						}

						AUTO(upgradedSession, onLowLevelRequestHeaders(m_requestHeaders, transferEncoding, contentLength));
						if(upgradedSession){
							const Mutex::UniqueLock lock(m_upgradedSessionMutex);
							m_upgradedSession = STD_MOVE(upgradedSession);
						}

						m_transferEncoding = STD_MOVE(transferEncoding);

						if(contentLength == CONTENT_CHUNKED){
							m_expectingNewLine = true;
							m_state = S_CHUNK_HEADER;
						} else {
							m_expectingNewLine = false;
							m_sizeExpecting = contentLength;
							m_state = S_IDENTITY;
						}
					}
					break;

				case S_IDENTITY:
					// epoll 线程读取 m_upgradedSession 不需要锁。
					if(m_upgradedSession){
						m_upgradedSession->onInit(STD_MOVE(m_requestHeaders),
							STD_MOVE(m_transferEncoding), STD_MOVE(expected));
					} else {
						onLowLevelRequest(STD_MOVE(m_requestHeaders),
							STD_MOVE(m_transferEncoding), STD_MOVE(expected));
					}

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
						m_requestHeaders.headers.append(
							SharedNts(line.c_str(), pos), line.substr(line.find_first_not_of(' ', pos + 1)));

						m_expectingNewLine = true;
						// m_state = S_CHUNKED_TRAILER;
					} else {
						// epoll 线程读取 m_upgradedSession 不需要锁。
						if(m_upgradedSession){
							m_upgradedSession->onInit(STD_MOVE(m_requestHeaders),
								STD_MOVE(m_transferEncoding), STD_MOVE(m_chunkedEntity));
						} else {
							onLowLevelRequest(STD_MOVE(m_requestHeaders),
								STD_MOVE(m_transferEncoding), STD_MOVE(m_chunkedEntity));
						}

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
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Http::Exception thrown while parsing data, URI = ", m_requestHeaders.uri, ", status = ", e.statusCode());
			onLowLevelError(e.statusCode(), e.headers());
			shutdownRead();
			shutdownWrite();
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"std::exception thrown while parsing data, URI = ", m_requestHeaders.uri, ", what = ", e.what());
			onLowLevelError(ST_INTERNAL_SERVER_ERROR, OptionalMap());
			shutdownRead();
			shutdownWrite();
		}
	}

	void LowLevelSession::onReadHup() NOEXCEPT {
		PROFILE_ME;

		// epoll 线程读取 m_upgradedSession 不需要锁。
		if(m_upgradedSession){
			m_upgradedSession->onReadHup();
		}

		TcpSessionBase::onReadHup();
	}
	void LowLevelSession::onWriteHup() NOEXCEPT {
		PROFILE_ME;

		// epoll 线程读取 m_upgradedSession 不需要锁。
		if(m_upgradedSession){
			m_upgradedSession->onWriteHup();
		}

		TcpSessionBase::onWriteHup();
	}
	void LowLevelSession::onClose(int errCode) NOEXCEPT {
		PROFILE_ME;

		// epoll 线程读取 m_upgradedSession 不需要锁。
		if(m_upgradedSession){
			m_upgradedSession->onClose(errCode);
		}

		TcpSessionBase::onClose(errCode);
	}

	boost::shared_ptr<UpgradedLowLevelSessionBase> LowLevelSession::getUpgradedSession() const {
		const Mutex::UniqueLock lock(m_upgradedSessionMutex);
		return m_upgradedSession;
	}

	bool LowLevelSession::send(ResponseHeaders responseHeaders, StreamBuffer entity){
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
		return TcpSessionBase::send(STD_MOVE(data));
	}

	bool LowLevelSession::send(StatusCode statusCode, OptionalMap headers, StreamBuffer entity){
		PROFILE_ME;
		LOG_POSEIDON_DEBUG("Making HTTP/1.1 response: statusCode = ", statusCode);

		ResponseHeaders responseHeaders;
		responseHeaders.version = 10001;
		responseHeaders.statusCode = statusCode;
		responseHeaders.reason = getStatusCodeDesc(statusCode).descShort;
		responseHeaders.headers = STD_MOVE(headers);
		return send(STD_MOVE(responseHeaders), STD_MOVE(entity));
	}

	bool LowLevelSession::sendDefault(StatusCode statusCode, OptionalMap headers){
		PROFILE_ME;
		LOG_POSEIDON_DEBUG("Making default HTTP/1.1 response: statusCode = ", statusCode);

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
		return send(statusCode, STD_MOVE(headers), STD_MOVE(entity));
	}
}

}
