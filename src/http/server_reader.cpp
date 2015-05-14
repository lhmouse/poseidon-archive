// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "server_reader.hpp"
#include "const_strings.hpp"
#include "exception.hpp"
#include "utilities.hpp"
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "../log.hpp"
#include "../profiler.hpp"
#include "../string.hpp"

namespace Poseidon {

namespace Http {
	ServerReader::ServerReader()
		: m_sizeExpecting(EXPECTING_NEW_LINE), m_state(S_FIRST_HEADER)
	{
	}
	ServerReader::~ServerReader(){
		if(m_state != S_FIRST_HEADER){
			LOG_POSEIDON_WARNING("Now that this reader is to be destroyed, a premature request has to be discarded.");
		}
	}

	bool ServerReader::putEncodedData(StreamBuffer encoded){
		PROFILE_ME;

		m_queue.splice(encoded);

		bool hasNextRequest = true;
		do {
			const bool expectingNewLine = (m_sizeExpecting == EXPECTING_NEW_LINE);

			if(expectingNewLine){
				struct Helper {
					static bool traverseCallback(void *ctx, const void *data, std::size_t size){
						AUTO_REF(lfOffset, *static_cast<std::size_t *>(ctx));

						const AUTO(begin, static_cast<const char *>(data));
						const AUTO(pos, static_cast<const char *>(std::memchr(begin, '\n', size)));
						if(!pos){
							lfOffset += size;
							return true;
						}
						lfOffset += static_cast<std::size_t>(pos - begin);
						return false;
					}
				};

				std::size_t lfOffset = 0;
				if(m_queue.traverse(&Helper::traverseCallback, &lfOffset)){
					// 没找到换行符。
					break;
				}
				// 找到了。
				m_sizeExpecting = lfOffset + 1;
			} else {
				if(m_queue.size() < m_sizeExpecting){
					break;
				}
			}

			AUTO(expected, m_queue.cut(m_sizeExpecting));
			if(expectingNewLine){
				expected.unput(); // '\n'
				if(expected.back() == '\r'){
					expected.unput();
				}
			}

			switch(m_state){
				boost::uint64_t temp64;

			case S_FIRST_HEADER:
				if(!expected.empty()){
					m_requestHeaders = RequestHeaders();
					m_contentLength = 0;
					m_contentOffset = 0;

					std::string line;
					expected.dump(line);

					AUTO(pos, line.find(' '));
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

					m_sizeExpecting = EXPECTING_NEW_LINE;
					m_state = S_HEADERS;
				} else {
					m_sizeExpecting = EXPECTING_NEW_LINE;
					// m_state = S_FIRST_HEADER;
				}
				break;

			case S_HEADERS:
				if(!expected.empty()){
					std::string line;
					expected.dump(line);

					AUTO(pos, line.find(':'));
					if(pos == std::string::npos){
						LOG_POSEIDON_WARNING("Invalid HTTP header: line = ", line);
						DEBUG_THROW(Exception, ST_BAD_REQUEST);
					}
					m_requestHeaders.headers.append(SharedNts(line.data(), pos), ltrim(line.substr(pos + 1)));

					m_sizeExpecting = EXPECTING_NEW_LINE;
					// m_state = S_HEADERS;
				} else {
					AUTO(transferEncoding, m_requestHeaders.headers.get("Transfer-Encoding"));
					AUTO(pos, transferEncoding.find(';'));
					if(pos != std::string::npos){
						transferEncoding.erase(pos);
					}
					transferEncoding = toLowerCase(trim(STD_MOVE(transferEncoding)));
					if(transferEncoding.empty() || (transferEncoding == STR_IDENTITY)){
						transferEncoding.clear();

						const AUTO_REF(contentLength, m_requestHeaders.headers.get("Content-Length"));
						if(contentLength.empty()){
							m_contentLength = 0;
						} else {
							char *endptr;
							m_contentLength = ::strtoull(contentLength.c_str(), &endptr, 10);
							if(*endptr){
								LOG_POSEIDON_WARNING("Bad request header Content-Length: ", contentLength);
								DEBUG_THROW(Exception, ST_BAD_REQUEST);
							}
							if(m_contentLength > CONTENT_LENGTH_MAX){
								LOG_POSEIDON_WARNING("Inacceptable Content-Length: ", contentLength);
								DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
							}
						}
					} else {
						m_contentLength = CONTENT_CHUNKED;
					}

					onRequestHeaders(STD_MOVE(m_requestHeaders), STD_MOVE(transferEncoding), m_contentLength);

					if(m_contentLength == CONTENT_CHUNKED){
						m_sizeExpecting = EXPECTING_NEW_LINE;
						m_state = S_CHUNK_HEADER;
					} else {
						m_sizeExpecting = std::min<boost::uint64_t>(m_contentLength, 1024);
						m_state = S_IDENTITY;
					}
				}
				break;

			case S_IDENTITY:
				temp64 = std::min<boost::uint64_t>(expected.size(), m_contentLength - m_contentOffset);
				onRequestEntity(m_contentOffset, expected.cut(temp64));
				m_contentOffset += temp64;

				if(m_contentOffset < m_contentLength){
					m_sizeExpecting = std::min<boost::uint64_t>(m_contentLength - m_contentOffset, 1024);
					// m_state = S_IDENTITY;
				} else {
					hasNextRequest = onRequestEnd(m_contentOffset, VAL_INIT);

					m_sizeExpecting = EXPECTING_NEW_LINE;
					m_state = S_FIRST_HEADER;
				}
				break;

			case S_CHUNK_HEADER:
				if(!expected.empty()){
					m_chunkSize = 0;
					m_chunkOffset = 0;
					m_chunkedTrailer.clear();

					std::string line;
					expected.dump(line);

					char *endptr;
					m_chunkSize = ::strtoull(line.c_str(), &endptr, 16);
					if(*endptr && (*endptr != ' ')){
						LOG_POSEIDON_WARNING("Bad chunk header: ", line);
						DEBUG_THROW(Exception, ST_BAD_REQUEST);
					}
					if(m_chunkSize > CONTENT_LENGTH_MAX){
						LOG_POSEIDON_WARNING("Inacceptable chunk size in header: ", line);
						DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
					}
					if(m_chunkSize == 0){
						m_sizeExpecting = EXPECTING_NEW_LINE;
						m_state = S_CHUNKED_TRAILER;
					} else {
						m_sizeExpecting = std::min<boost::uint64_t>(m_chunkSize, 1024);
						m_state = S_CHUNK_DATA;
					}
				} else {
					// chunk-data 后面应该有一对 CRLF。我们在这里处理这种情况。
					m_sizeExpecting = EXPECTING_NEW_LINE;
					// m_state = S_CHUNK_HEADER;
				}
				break;

			case S_CHUNK_DATA:
				temp64 = std::min<boost::uint64_t>(expected.size(), m_chunkSize - m_chunkOffset);
				onRequestEntity(m_contentOffset, expected.cut(temp64));
				m_contentOffset += temp64;
				m_chunkOffset += temp64;

				if(m_chunkOffset < m_chunkSize){
					m_sizeExpecting = std::min<boost::uint64_t>(m_chunkSize - m_chunkOffset, 1024);
					// m_state = S_CHUNK_DATA;
				} else {
					m_sizeExpecting = EXPECTING_NEW_LINE;
					m_state = S_CHUNK_HEADER;
				}
				break;

			case S_CHUNKED_TRAILER:
				if(!expected.empty()){
					std::string line;
					expected.dump(line);

					AUTO(pos, line.find(':'));
					if(pos == std::string::npos){
						LOG_POSEIDON_WARNING("Invalid chunk trailer: line = ", line);
						DEBUG_THROW(Exception, ST_BAD_REQUEST);
					}
					m_chunkedTrailer.append(SharedNts(line.data(), pos), ltrim(line.substr(pos + 1)));

					m_sizeExpecting = EXPECTING_NEW_LINE;
					// m_state = S_CHUNKED_TRAILER;
				} else {
					hasNextRequest = onRequestEnd(m_contentOffset, STD_MOVE(m_chunkedTrailer));

					m_sizeExpecting = EXPECTING_NEW_LINE;
					m_state = S_FIRST_HEADER;
				}
				break;

			default:
				LOG_POSEIDON_ERROR("Unknown state: ", static_cast<unsigned>(m_state));
				std::abort();
			}
		} while(hasNextRequest);

		return hasNextRequest;
	}
}

}
