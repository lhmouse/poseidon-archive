// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "server_reader.hpp"
#include "exception.hpp"
#include "urlencoded.hpp"
#include <sys/types.h>
#include <unistd.h>
#include "../log.hpp"
#include "../profiler.hpp"
#include "../string.hpp"
#include "../singletons/main_config.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {
namespace Http {

Server_reader::Server_reader()
	: m_size_expecting(content_length_expecting_endl), m_state(state_first_header)
{
	//
}
Server_reader::~Server_reader(){
	if(m_state != state_first_header){
		POSEIDON_LOG_DEBUG("Now that this reader is to be destroyed, a premature request has to be discarded.");
	}
}

bool Server_reader::put_encoded_data(Stream_buffer encoded, bool dont_parse_get_params){
	POSEIDON_PROFILE_ME;

	m_queue.splice(encoded);

	bool has_next_request = true;
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
				const AUTO(max_line_length, Main_config::get<std::size_t>("http_max_header_line_length", 8192));
				POSEIDON_THROW_UNLESS(m_queue.size() <= max_line_length, Exception, status_bad_request); // XXX 用一个别的状态码？
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
				m_request_headers = Request_headers();
				m_content_length = 0;
				m_content_offset = 0;

				std::string line = expected.dump_string();

				for(AUTO(it, line.begin()); it != line.end(); ++it){
					const unsigned ch = static_cast<unsigned char>(*it);
					POSEIDON_THROW_UNLESS((0x20 <= ch) && (ch <= 0x7E), Basic_exception, Rcnts::view("Invalid HTTP request header"));
				}

				AUTO(pos, line.find(' '));
				POSEIDON_THROW_UNLESS(pos != std::string::npos, Exception, status_bad_request);
				line.at(pos) = 0;
				m_request_headers.verb = get_verb_from_string(line.c_str());
				POSEIDON_THROW_UNLESS(m_request_headers.verb != verb_invalid, Exception, status_not_implemented);
				line.erase(0, pos + 1);

				pos = line.find(' ');
				POSEIDON_THROW_UNLESS(pos != std::string::npos, Exception, status_bad_request);
				m_request_headers.uri.assign(line, 0, pos);
				line.erase(0, pos + 1);

				long ver_end = 0;
				char ver_major_str[16], ver_minor_str[16];
				POSEIDON_THROW_UNLESS(std::sscanf(line.c_str(), "HTTP/%15[0-9].%15[0-9]%ln", ver_major_str, ver_minor_str, &ver_end) == 2, Exception, status_bad_request);
				POSEIDON_THROW_UNLESS(static_cast<std::size_t>(ver_end) == line.size(), Exception, status_bad_request);
				m_request_headers.version = boost::numeric_cast<unsigned>(std::strtoul(ver_major_str, NULLPTR, 10) * 10000 + std::strtoul(ver_minor_str, NULLPTR, 10));
				POSEIDON_THROW_UNLESS(m_request_headers.version <= 10001, Exception, status_version_not_supported);

				if(!dont_parse_get_params){
					pos = m_request_headers.uri.find('?');
					if(pos != std::string::npos){
						Buffer_istream is;
						is.set_buffer(Stream_buffer(m_request_headers.uri.data() + pos + 1, m_request_headers.uri.size() - pos - 1));
						url_decode_params(is, m_request_headers.get_params);
						m_request_headers.uri.erase(pos);
					}
				}

				m_size_expecting = content_length_expecting_endl;
				m_state = state_headers;
			} else {
				m_size_expecting = content_length_expecting_endl;
				// m_state = state_first_header;
			}
			break;

		case state_headers:
			if(!expected.empty()){
				const AUTO(headers, m_request_headers.headers.size());
				const AUTO(max_headers, Main_config::get<std::size_t>("http_max_headers_per_request", 64));
				POSEIDON_THROW_UNLESS(headers <= max_headers, Exception, status_bad_request); // XXX 用一个别的状态码？

				std::string line = expected.dump_string();

				AUTO(pos, line.find(':'));
				POSEIDON_THROW_UNLESS(pos != std::string::npos, Exception, status_bad_request);
				Rcnts key(line.data(), pos);
				line.erase(0, pos + 1);
				std::string value(trim(STD_MOVE(line)));
				m_request_headers.headers.append(STD_MOVE(key), STD_MOVE(value));

				m_size_expecting = content_length_expecting_endl;
				// m_state = state_headers;
			} else {
				const AUTO_REF(transfer_encoding, m_request_headers.headers.get("Transfer-Encoding"));
				if(transfer_encoding.empty() || (::strcasecmp(transfer_encoding.c_str(), "identity") == 0)){
					const AUTO_REF(content_length, m_request_headers.headers.get("Content-Length"));
					if(content_length.empty()){
						m_content_length = 0;
					} else {
						char *eptr;
						m_content_length = std::strtoull(content_length.c_str(), &eptr, 10);
						POSEIDON_THROW_UNLESS(*eptr == 0, Exception, status_bad_request);
						POSEIDON_THROW_UNLESS(m_content_length <= content_length_max, Exception, status_payload_too_large);
					}
				} else if(::strcasecmp(transfer_encoding.c_str(), "chunked") == 0){
					m_content_length = content_length_chunked;
				} else {
					POSEIDON_LOG_WARNING("Inacceptable Transfer-Encoding: ", transfer_encoding);
					POSEIDON_THROW(Basic_exception, Rcnts::view("Inacceptable Transfer-Encoding"));
				}

				on_request_headers(STD_MOVE(m_request_headers), m_content_length);

				if(m_content_length == content_length_chunked){
					m_size_expecting = content_length_expecting_endl;
					m_state = state_chunk_header;
				} else {
					m_size_expecting = std::min<std::uint64_t>(m_content_length, 4096);
					m_state = state_identity;
				}
			}
			break;

		case state_identity:
			temp64 = std::min<std::uint64_t>(expected.size(), m_content_length - m_content_offset);
			if(temp64 > 0){
				on_request_entity(m_content_offset, expected.cut_off(boost::numeric_cast<std::size_t>(temp64)));
			}
			m_content_offset += temp64;

			if(m_content_offset < m_content_length){
				m_size_expecting = std::min<std::uint64_t>(m_content_length - m_content_offset, 4096);
				// m_state = state_identity;
			} else {
				has_next_request = on_request_end(m_content_offset, VAL_INIT);

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
				POSEIDON_THROW_UNLESS((*eptr == 0) || (*eptr == ' '), Exception, status_bad_request);
				POSEIDON_THROW_UNLESS(m_chunk_size <= content_length_max, Exception, status_payload_too_large);
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
			assert(temp64 > 0);
			on_request_entity(m_content_offset, expected.cut_off(boost::numeric_cast<std::size_t>(temp64)));
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
				POSEIDON_THROW_UNLESS(pos != std::string::npos, Exception, status_bad_request);
				Rcnts key(line.data(), pos);
				line.erase(0, pos + 1);
				std::string value(trim(STD_MOVE(line)));
				m_chunked_trailer.append(STD_MOVE(key), STD_MOVE(value));

				m_size_expecting = content_length_expecting_endl;
				// m_state = state_chunked_trailer;
			} else {
				has_next_request = on_request_end(m_content_offset, STD_MOVE(m_chunked_trailer));

				m_size_expecting = content_length_expecting_endl;
				m_state = state_first_header;
			}
			break;
		}
	} while(has_next_request);

	return has_next_request;
}

}
}
