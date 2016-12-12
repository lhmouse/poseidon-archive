// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client_reader.hpp"
#include "exception.hpp"
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "../log.hpp"
#include "../profiler.hpp"
#include "../string.hpp"

namespace Poseidon {

namespace Http {
	ClientReader::ClientReader()
		: m_size_expecting(EXPECTING_NEW_LINE), m_state(S_FIRST_HEADER)
	{
	}
	ClientReader::~ClientReader(){
		if(m_state != S_FIRST_HEADER){
			LOG_POSEIDON_DEBUG("Now that this reader is to be destroyed, a premature response has to be discarded.");
		}
	}

	bool ClientReader::put_encoded_data(StreamBuffer encoded){
		PROFILE_ME;

		m_queue.splice(encoded);

		bool has_next_response = true;
		do {
			const bool expecting_new_line = (m_size_expecting == EXPECTING_NEW_LINE);

			if(expecting_new_line){
				std::size_t lf_offset = 0;
				AUTO(ce, m_queue.get_const_chunk_enumerator());
				for(;;){
					if(!ce){
						lf_offset = static_cast<std::size_t>(-1);
						break;
					}
					const AUTO(pos, std::find(ce.begin(), ce.end(), '\n'));
					if(pos != ce.end()){
						lf_offset += static_cast<std::size_t>(pos - ce.begin());
						break;
					}
					lf_offset += ce.size();
					++ce;
				}
				if(lf_offset == static_cast<std::size_t>(-1)){
					// 没找到换行符。
					break;
				}
				m_size_expecting = lf_offset + 1;
			} else {
				if(m_queue.size() < m_size_expecting){
					break;
				}
			}

			AUTO(expected, m_queue.cut_off(m_size_expecting));
			if(expecting_new_line){
				expected.unput(); // '\n'
				if(expected.back() == '\r'){
					expected.unput();
				}
			}

			switch(m_state){
				boost::uint64_t temp64;

			case S_FIRST_HEADER:
				if(!expected.empty()){
					m_response_headers = ResponseHeaders();
					m_content_length = 0;
					m_content_offset = 0;

					std::string line = expected.dump_string();

					AUTO(pos, line.find(' '));
					if(pos == std::string::npos){
						LOG_POSEIDON_WARNING("Bad request header: expecting verb, line = ", line);
						DEBUG_THROW(BasicException, sslit("No HTTP version in response headers"));
					}
					line[pos] = 0;
					long ver_end = 0;
					char ver_major_str[16], ver_minor_str[16];
					if(std::sscanf(line.c_str(), "HTTP/%15[0-9].%15[0-9]%ln", ver_major_str, ver_minor_str, &ver_end) != 2){
						LOG_POSEIDON_WARNING("Bad response header: expecting HTTP version:", line);
						DEBUG_THROW(BasicException, sslit("Malformed HTTP version in response headers"));
					}
					if(static_cast<unsigned long>(ver_end) != pos){
						LOG_POSEIDON_WARNING("Bad response header: junk after HTTP version:", line);
						DEBUG_THROW(BasicException, sslit("Malformed HTTP version in response headers"));
					}
					m_response_headers.version = std::strtoul(ver_major_str, NULLPTR, 10) * 10000 + std::strtoul(ver_minor_str, NULLPTR, 10);
					line.erase(0, pos + 1);

					pos = line.find(' ');
					if(pos == std::string::npos){
						LOG_POSEIDON_WARNING("Bad response header: expecting status code:", line);
						DEBUG_THROW(BasicException, sslit("No status code in response headers"));
					}
					line[pos] = 0;
					char *endptr;
					const AUTO(status_code, std::strtoul(line.c_str(), &endptr, 10));
					if(*endptr){
						LOG_POSEIDON_WARNING("Bad response header: expecting status code:", line);
						DEBUG_THROW(BasicException, sslit("Malformed status code in response headers"));
					}
					m_response_headers.status_code = status_code;
					line.erase(0, pos + 1);

					m_response_headers.reason = STD_MOVE(line);

					m_size_expecting = EXPECTING_NEW_LINE;
					m_state = S_HEADERS;
				} else {
					m_size_expecting = EXPECTING_NEW_LINE;
					// m_state = S_FIRST_HEADER;
				}
				break;

			case S_HEADERS:
				if(!expected.empty()){
					std::string line = expected.dump_string();

					AUTO(pos, line.find(':'));
					if(pos == std::string::npos){
						LOG_POSEIDON_WARNING("Invalid HTTP header: ", line);
						DEBUG_THROW(BasicException, sslit("Malformed HTTP header in response headers"));
					}
					SharedNts key(line.data(), pos);
					line.erase(0, pos + 1);
					std::string value(ltrim(STD_MOVE(line)));
					m_response_headers.headers.append(STD_MOVE(key), STD_MOVE(value));

					m_size_expecting = EXPECTING_NEW_LINE;
					// m_state = S_HEADERS;
				} else {
					const AUTO_REF(transfer_encoding, m_response_headers.headers.get("Transfer-Encoding"));
					if(transfer_encoding.empty() || (::strcasecmp(transfer_encoding.c_str(), "identity") == 0)){
						const AUTO_REF(content_length, m_response_headers.headers.get("Content-Length"));
						if(content_length.empty()){
							m_content_length = CONTENT_TILL_EOF;
						} else {
							char *endptr;
							m_content_length = ::strtoull(content_length.c_str(), &endptr, 10);
							if(*endptr){
								LOG_POSEIDON_WARNING("Bad request header Content-Length: ", content_length);
								DEBUG_THROW(BasicException, sslit("Malformed Content-Length header"));
							}
							if(m_content_length > CONTENT_LENGTH_MAX){
								LOG_POSEIDON_WARNING("Inacceptable Content-Length: ", content_length);
								DEBUG_THROW(BasicException, sslit("Inacceptable Content-Length"));
							}
						}
					} else if(::strcasecmp(transfer_encoding.c_str(), "chunked") == 0){
						m_content_length = CONTENT_CHUNKED;
					} else {
						LOG_POSEIDON_WARNING("Inacceptable Transfer-Encoding: ", transfer_encoding);
						DEBUG_THROW(BasicException, sslit("Inacceptable Transfer-Encoding"));
					}

					on_response_headers(STD_MOVE(m_response_headers), m_content_length);

					if(m_content_length == CONTENT_CHUNKED){
						m_size_expecting = EXPECTING_NEW_LINE;
						m_state = S_CHUNK_HEADER;
					} else if(m_content_length == CONTENT_TILL_EOF){
						m_size_expecting = 4096;
						m_state = S_IDENTITY;
					} else {
						m_size_expecting = std::min<boost::uint64_t>(m_content_length, 4096);
						m_state = S_IDENTITY;
					}
				}
				break;

			case S_IDENTITY:
				temp64 = std::min<boost::uint64_t>(expected.size(), m_content_length - m_content_offset);
				on_response_entity(m_content_offset, expected.cut_off(temp64));
				m_content_offset += temp64;

				if(m_content_length == CONTENT_TILL_EOF){
					m_size_expecting = 4096;
					// m_state = S_IDENTITY;
				} else if(m_content_offset < m_content_length){
					m_size_expecting = std::min<boost::uint64_t>(m_content_length - m_content_offset, 4096);
					// m_state = S_IDENTITY;
				} else {
					has_next_response = on_response_end(m_content_offset, VAL_INIT);

					m_size_expecting = EXPECTING_NEW_LINE;
					m_state = S_FIRST_HEADER;
				}
				break;

			case S_CHUNK_HEADER:
				if(!expected.empty()){
					m_chunk_size = 0;
					m_chunk_offset = 0;
					m_chunked_trailer.clear();

					std::string line = expected.dump_string();

					char *endptr;
					m_chunk_size = ::strtoull(line.c_str(), &endptr, 16);
					if(*endptr && (*endptr != ' ')){
						LOG_POSEIDON_WARNING("Bad chunk header: ", line);
						DEBUG_THROW(BasicException, sslit("Malformed chunk header"));
					}
					if(m_chunk_size > CONTENT_LENGTH_MAX){
						LOG_POSEIDON_WARNING("Inacceptable chunk size in header: ", line);
						DEBUG_THROW(BasicException, sslit("Inacceptable chunk length"));
					}
					if(m_chunk_size == 0){
						m_size_expecting = EXPECTING_NEW_LINE;
						m_state = S_CHUNKED_TRAILER;
					} else {
						m_size_expecting = std::min<boost::uint64_t>(m_chunk_size, 4096);
						m_state = S_CHUNK_DATA;
					}
				} else {
					// chunk-data 后面应该有一对 CRLF。我们在这里处理这种情况。
					m_size_expecting = EXPECTING_NEW_LINE;
					// m_state = S_CHUNK_HEADER;
				}
				break;

			case S_CHUNK_DATA:
				temp64 = std::min<boost::uint64_t>(expected.size(), m_chunk_size - m_chunk_offset);
				on_response_entity(m_content_offset, expected.cut_off(temp64));
				m_content_offset += temp64;
				m_chunk_offset += temp64;

				if(m_chunk_offset < m_chunk_size){
					m_size_expecting = std::min<boost::uint64_t>(m_chunk_size - m_chunk_offset, 4096);
					// m_state = S_CHUNK_DATA;
				} else {
					m_size_expecting = EXPECTING_NEW_LINE;
					m_state = S_CHUNK_HEADER;
				}
				break;

			case S_CHUNKED_TRAILER:
				if(!expected.empty()){
					std::string line = expected.dump_string();

					AUTO(pos, line.find(':'));
					if(pos == std::string::npos){
						LOG_POSEIDON_WARNING("Invalid chunk trailer:", line);
						DEBUG_THROW(BasicException, sslit("Invalid HTTP header in chunk trailer"));
					}
					SharedNts key(line.data(), pos);
					line.erase(0, pos + 1);
					std::string value(ltrim(STD_MOVE(line)));
					m_chunked_trailer.append(STD_MOVE(key), STD_MOVE(value));

					m_size_expecting = EXPECTING_NEW_LINE;
					// m_state = S_CHUNKED_TRAILER;
				} else {
					has_next_response = on_response_end(m_content_offset, STD_MOVE(m_chunked_trailer));

					m_size_expecting = EXPECTING_NEW_LINE;
					m_state = S_FIRST_HEADER;
				}
				break;
			}
		} while(has_next_response);

		return has_next_response;
	}

	bool ClientReader::is_content_till_eof() const {
		if(m_state < S_IDENTITY){
			return false;
		}
		return m_content_length == CONTENT_TILL_EOF;
	}
	bool ClientReader::terminate_content(){
		PROFILE_ME;

		if(!is_content_till_eof()){
			DEBUG_THROW(BasicException, sslit("Terminating a non-until-EOF HTTP response"));
		}

		const AUTO(bytes_remaining, m_queue.size());
		if(bytes_remaining != 0){
			on_response_entity(m_content_offset, STD_MOVE(m_queue));
			m_content_offset += bytes_remaining;
			m_queue.clear();
		}

		const bool ret = on_response_end(m_content_offset, STD_MOVE(m_chunked_trailer));

		m_size_expecting = EXPECTING_NEW_LINE;
		m_state = S_FIRST_HEADER;

		return ret;
	}
}

}
