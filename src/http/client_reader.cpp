// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client_reader.hpp"
#include "exception.hpp"
#include <sys/types.h>
#include <unistd.h>
#include "../log.hpp"
#include "../profiler.hpp"
#include "../string.hpp"

namespace Poseidon {
namespace Http {

ClientReader::ClientReader()
	: m_size_expecting(EXPECTING_NEW_LINE), m_state(S_FIRST_HEADER)
{ }
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
			std::ptrdiff_t lf_offset = 0;
			StreamBuffer::EnumerationCookie cookie;
			for(;;){
				const void *data, *pos;
				std::size_t size;
				if(!m_queue.enumerate_chunk(&data, &size, cookie)){
					lf_offset = -1;
					break;
				}
				pos = std::memchr(data, '\n', size);
				if(pos){
					lf_offset += static_cast<const char *>(pos) - static_cast<const char *>(data);
					break;
				}
				lf_offset += static_cast<std::ptrdiff_t>(size);
			}
			if(lf_offset < 0){
				// 没找到换行符。
				break;
			}
			m_size_expecting = static_cast<std::size_t>(lf_offset) + 1;
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

				for(AUTO(it, line.begin()); it != line.end(); ++it){
					const unsigned ch = static_cast<unsigned char>(*it);
					DEBUG_THROW_UNLESS((0x20 <= ch) && (ch <= 0x7E), BasicException, sslit("Invalid HTTP response header"));
				}

				AUTO(pos, line.find(' '));
				DEBUG_THROW_UNLESS(pos != std::string::npos, BasicException, sslit("No HTTP version in response headers"));
				line.at(pos) = 0;
				long ver_end = 0;
				char ver_major_str[16], ver_minor_str[16];
				DEBUG_THROW_UNLESS(std::sscanf(line.c_str(), "HTTP/%15[0-9].%15[0-9]%ln", ver_major_str, ver_minor_str, &ver_end) == 2, BasicException, sslit("Malformed HTTP version in response headers"));
				DEBUG_THROW_UNLESS(static_cast<std::size_t>(ver_end) == pos, BasicException, sslit("Malformed HTTP version in response headers"));
				m_response_headers.version = boost::numeric_cast<unsigned>(std::strtoul(ver_major_str, NULLPTR, 10) * 10000 + std::strtoul(ver_minor_str, NULLPTR, 10));
				line.erase(0, pos + 1);

				pos = line.find(' ');
				DEBUG_THROW_UNLESS(pos != std::string::npos, BasicException, sslit("No status code in response headers"));
				line[pos] = 0;
				char *eptr;
				const unsigned status_code = boost::numeric_cast<unsigned>(std::strtoul(line.c_str(), &eptr, 10));
				DEBUG_THROW_UNLESS(*eptr == 0, BasicException, sslit("Malformed status code in response headers"));
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
				DEBUG_THROW_UNLESS(pos != std::string::npos, BasicException, sslit("Malformed HTTP header in response headers"));
				SharedNts key(line.data(), pos);
				line.erase(0, pos + 1);
				std::string value(trim(STD_MOVE(line)));
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
						char *eptr;
						m_content_length = ::strtoull(content_length.c_str(), &eptr, 10);
						DEBUG_THROW_UNLESS(*eptr == 0, BasicException, sslit("Malformed Content-Length header"));
						DEBUG_THROW_UNLESS(m_content_length <= CONTENT_LENGTH_MAX, BasicException, sslit("Inacceptable Content-Length"));
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

				char *eptr;
				m_chunk_size = ::strtoull(line.c_str(), &eptr, 16);
				DEBUG_THROW_UNLESS((*eptr == 0) || (*eptr == ' '), BasicException, sslit("Malformed chunk header"));
				DEBUG_THROW_UNLESS(m_chunk_size <= CONTENT_LENGTH_MAX, BasicException, sslit("Inacceptable chunk length"));
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
				DEBUG_THROW_UNLESS(pos != std::string::npos, BasicException, sslit("Invalid HTTP header in chunk trailer"));
				SharedNts key(line.data(), pos);
				line.erase(0, pos + 1);
				std::string value(trim(STD_MOVE(line)));
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
	DEBUG_THROW_ASSERT(is_content_till_eof());

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
