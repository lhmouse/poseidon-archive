// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

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

Client_reader::Client_reader()
	: m_size_expecting(content_length_expecting_endl), m_state(state_first_header)
{
	//
}
Client_reader::~Client_reader(){
	if(m_state != state_first_header){
		POSEIDON_LOG_DEBUG("Now that this reader is to be destroyed, a premature response has to be discarded.");
	}
}

bool Client_reader::put_encoded_data(Stream_buffer encoded){
	POSEIDON_PROFILE_ME;

	m_queue.splice(encoded);

	bool has_next_response = true;
	do {
		const bool expecting_new_line = (m_size_expecting == content_length_expecting_endl);

		if(expecting_new_line){
			std::ptrdiff_t lf_offset = 0;
			Stream_buffer::Enumeration_cookie cookie;
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

		AUTO(expected, m_queue.cut_off(boost::numeric_cast<std::size_t>(m_size_expecting)));
		if(expecting_new_line){
			expected.unput(); // '\n'
			if(expected.back() == '\r'){
				expected.unput();
			}
		}

		switch(m_state){
			std::uint64_t temp64;

		case state_first_header:
			if(!expected.empty()){
				m_response_headers = Response_headers();
				m_content_length = 0;
				m_content_offset = 0;

				std::string line = expected.dump_string();

				for(AUTO(it, line.begin()); it != line.end(); ++it){
					const unsigned ch = static_cast<unsigned char>(*it);
					POSEIDON_THROW_UNLESS((0x20 <= ch) && (ch <= 0x7E), Basic_exception, Rcnts::view("Invalid HTTP response header"));
				}

				AUTO(pos, line.find(' '));
				POSEIDON_THROW_UNLESS(pos != std::string::npos, Basic_exception, Rcnts::view("No HTTP version in response headers"));
				line.at(pos) = 0;
				long ver_end = 0;
				char ver_major_str[16], ver_minor_str[16];
				const int n_got = std::sscanf(line.c_str(), "HTTP/%15[0-9].%15[0-9]%ln", ver_major_str, ver_minor_str, &ver_end);
				POSEIDON_THROW_UNLESS(n_got == 2, Basic_exception, Rcnts::view("Malformed HTTP version in response headers"));
				POSEIDON_THROW_UNLESS(static_cast<std::size_t>(ver_end) == pos, Basic_exception, Rcnts::view("Malformed HTTP version in response headers"));
				m_response_headers.version = boost::numeric_cast<unsigned>(std::strtoul(ver_major_str, NULLPTR, 10) * 10000 + std::strtoul(ver_minor_str, NULLPTR, 10));
				line.erase(0, pos + 1);

				pos = line.find(' ');
				POSEIDON_THROW_UNLESS(pos != std::string::npos, Basic_exception, Rcnts::view("No status code in response headers"));
				line[pos] = 0;
				char *eptr;
				const auto status_code = boost::numeric_cast<int>(std::strtoul(line.c_str(), &eptr, 10));
				POSEIDON_THROW_UNLESS(*eptr == 0, Basic_exception, Rcnts::view("Malformed status code in response headers"));
				m_response_headers.status_code = status_code;
				line.erase(0, pos + 1);

				m_response_headers.reason = STD_MOVE(line);

				m_size_expecting = content_length_expecting_endl;
				m_state = state_headers;
			} else {
				m_size_expecting = content_length_expecting_endl;
				// m_state = state_first_header;
			}
			break;

		case state_headers:
			if(!expected.empty()){
				std::string line = expected.dump_string();

				AUTO(pos, line.find(':'));
				POSEIDON_THROW_UNLESS(pos != std::string::npos, Basic_exception, Rcnts::view("Malformed HTTP header in response headers"));
				Rcnts key(line.data(), pos);
				line.erase(0, pos + 1);
				std::string value(trim(STD_MOVE(line)));
				m_response_headers.headers.append(STD_MOVE(key), STD_MOVE(value));

				m_size_expecting = content_length_expecting_endl;
				// m_state = state_headers;
			} else {
				const AUTO_REF(transfer_encoding, m_response_headers.headers.get("Transfer-Encoding"));
				if(transfer_encoding.empty() || (::strcasecmp(transfer_encoding.c_str(), "identity") == 0)){
					const AUTO_REF(content_length, m_response_headers.headers.get("Content-Length"));
					if(content_length.empty()){
						m_content_length = content_length_until_eof;
					} else {
						char *eptr;
						m_content_length = std::strtoull(content_length.c_str(), &eptr, 10);
						POSEIDON_THROW_UNLESS(*eptr == 0, Basic_exception, Rcnts::view("Malformed Content-Length header"));
						POSEIDON_THROW_UNLESS(m_content_length <= content_length_max, Basic_exception, Rcnts::view("Inacceptable Content-Length"));
					}
				} else if(::strcasecmp(transfer_encoding.c_str(), "chunked") == 0){
					m_content_length = content_length_chunked;
				} else {
					POSEIDON_LOG_WARNING("Inacceptable Transfer-Encoding: ", transfer_encoding);
					POSEIDON_THROW(Basic_exception, Rcnts::view("Inacceptable Transfer-Encoding"));
				}

				on_response_headers(STD_MOVE(m_response_headers), m_content_length);

				if(m_content_length == content_length_chunked){
					m_size_expecting = content_length_expecting_endl;
					m_state = state_chunk_header;
				} else if(m_content_length == content_length_until_eof){
					m_size_expecting = 4096;
					m_state = state_identity;
				} else {
					m_size_expecting = std::min<std::uint64_t>(m_content_length, 4096);
					m_state = state_identity;
				}
			}
			break;

		case state_identity:
			temp64 = std::min<std::uint64_t>(expected.size(), m_content_length - m_content_offset);
			if(temp64 > 0){
				on_response_entity(m_content_offset, expected.cut_off(boost::numeric_cast<std::size_t>(temp64)));
			}
			m_content_offset += temp64;

			if(m_content_length == content_length_until_eof){
				m_size_expecting = 4096;
				// m_state = state_identity;
			} else if(m_content_offset < m_content_length){
				m_size_expecting = std::min<std::uint64_t>(m_content_length - m_content_offset, 4096);
				// m_state = state_identity;
			} else {
				has_next_response = on_response_end(m_content_offset, VAL_INIT);

				m_size_expecting = content_length_expecting_endl;
				m_state = state_first_header;
			}
			break;

		case state_chunk_header:
			if(!expected.empty()){
				m_chunk_size = 0;
				m_chunk_offset = 0;
				m_chunked_trailer.clear();

				std::string line = expected.dump_string();

				char *eptr;
				m_chunk_size = std::strtoull(line.c_str(), &eptr, 16);
				POSEIDON_THROW_UNLESS((*eptr == 0) || (*eptr == ' '), Basic_exception, Rcnts::view("Malformed chunk header"));
				POSEIDON_THROW_UNLESS(m_chunk_size <= content_length_max, Basic_exception, Rcnts::view("Inacceptable chunk length"));
				if(m_chunk_size == 0){
					m_size_expecting = content_length_expecting_endl;
					m_state = state_chunked_trailer;
				} else {
					m_size_expecting = std::min<std::uint64_t>(m_chunk_size, 4096);
					m_state = state_chunk_data;
				}
			} else {
				// chunk-data 后面应该有一对 CRLF。我们在这里处理这种情况。
				m_size_expecting = content_length_expecting_endl;
				// m_state = state_chunk_header;
			}
			break;

		case state_chunk_data:
			temp64 = std::min<std::uint64_t>(expected.size(), m_chunk_size - m_chunk_offset);
			on_response_entity(m_content_offset, expected.cut_off(boost::numeric_cast<std::size_t>(temp64)));
			m_content_offset += temp64;
			m_chunk_offset += temp64;

			if(m_chunk_offset < m_chunk_size){
				m_size_expecting = std::min<std::uint64_t>(m_chunk_size - m_chunk_offset, 4096);
				// m_state = state_chunk_data;
			} else {
				m_size_expecting = content_length_expecting_endl;
				m_state = state_chunk_header;
			}
			break;

		case state_chunked_trailer:
			if(!expected.empty()){
				std::string line = expected.dump_string();

				AUTO(pos, line.find(':'));
				POSEIDON_THROW_UNLESS(pos != std::string::npos, Basic_exception, Rcnts::view("Invalid HTTP header in chunk trailer"));
				Rcnts key(line.data(), pos);
				line.erase(0, pos + 1);
				std::string value(trim(STD_MOVE(line)));
				m_chunked_trailer.append(STD_MOVE(key), STD_MOVE(value));

				m_size_expecting = content_length_expecting_endl;
				// m_state = state_chunked_trailer;
			} else {
				has_next_response = on_response_end(m_content_offset, STD_MOVE(m_chunked_trailer));

				m_size_expecting = content_length_expecting_endl;
				m_state = state_first_header;
			}
			break;
		}
	} while(has_next_response);

	return has_next_response;
}

bool Client_reader::is_content_till_eof() const {
	if(m_state < state_identity){
		return false;
	}
	return m_content_length == content_length_until_eof;
}
bool Client_reader::terminate_content(){
	POSEIDON_PROFILE_ME;
	POSEIDON_THROW_ASSERT(is_content_till_eof());

	const AUTO(bytes_remaining, m_queue.size());
	if(bytes_remaining != 0){
		on_response_entity(m_content_offset, STD_MOVE(m_queue));
		m_content_offset += bytes_remaining;
		m_queue.clear();
	}

	const bool ret = on_response_end(m_content_offset, STD_MOVE(m_chunked_trailer));

	m_size_expecting = content_length_expecting_endl;
	m_state = state_first_header;

	return ret;
}

}
}
