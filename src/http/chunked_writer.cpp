// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "chunked_writer.hpp"
#include "exception.hpp"
#include "../tcp_session_base.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../string.hpp"

namespace Poseidon {

namespace Http {
	ChunkedWriter::ChunkedWriter(){
	}
	ChunkedWriter::ChunkedWriter(boost::shared_ptr<TcpSessionBase> session, StatusCode statusCode,
		OptionalMap headers, const std::vector<std::string> &transferEncoding)
	{
		reset(STD_MOVE(session), statusCode, STD_MOVE(headers), transferEncoding);
	}
	ChunkedWriter::ChunkedWriter(boost::shared_ptr<TcpSessionBase> session, StatusCode statusCode,
		const std::vector<std::string> &transferEncoding)
	{
		reset(STD_MOVE(session), statusCode, transferEncoding);
	}
	ChunkedWriter::~ChunkedWriter(){
		reset();
	}

	void ChunkedWriter::reset() NOEXCEPT {
		PROFILE_ME;

		if(!m_session){
			return;
		}

		LOG_POSEIDON_DEBUG("ChunkedWriter is not finalized, shutting down session...");
		m_session->forceShutdown();

		m_session.reset();
	}
	void ChunkedWriter::reset(boost::shared_ptr<TcpSessionBase> session, StatusCode statusCode,
		OptionalMap headers, const std::vector<std::string> &transferEncoding)
	{
		PROFILE_ME;

		if(m_session != session){
			reset();
		}
		if(!session){
			return;
		}

		StreamBuffer data;

		char first[64];
		unsigned len = (unsigned)std::sprintf(first, "HTTP/1.1 %u ", static_cast<unsigned>(statusCode));
		data.put(first, len);
		const AUTO(desc, getStatusCodeDesc(statusCode));
		data.put(desc.descShort);
		data.put("\r\n");

		headers.erase("Content-Length");
		if(transferEncoding.empty()){
			headers.set("Transfer-Encoding", "chunked");
		} else {
			headers.set("Transfer-Encoding", implode(',', transferEncoding));
		}
		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			if(it->second.empty()){
				continue;
			}
			data.put(it->first.get());
			data.put(": ");
			data.put(it->second.data(), it->second.size());
			data.put("\r\n");
		}
		data.put("\r\n");

		session->send(STD_MOVE(data));

		m_session = STD_MOVE(session); // noexcept
	}
	void ChunkedWriter::reset(boost::shared_ptr<TcpSessionBase> session, StatusCode statusCode,
		const std::vector<std::string> &transferEncoding)
	{
		PROFILE_ME;

		reset(STD_MOVE(session), statusCode, OptionalMap(), transferEncoding);
	}

	void ChunkedWriter::put(StreamBuffer buffer){
		PROFILE_ME;

		if(buffer.empty()){
			return;
		}

		StreamBuffer data;

		char temp[64];
		unsigned len = (unsigned)std::sprintf(temp, "%llx\r\n", static_cast<unsigned long long>(buffer.size()));
		data.put(temp, len);
		data.splice(buffer);
		data.put("\r\n");

		m_session->send(STD_MOVE(data));
	}

	void ChunkedWriter::finalize(OptionalMap headers){
		if(!m_session){
			return;
		}

		StreamBuffer data;

		data.put("0\r\n");

		headers.erase("Content-Length");
		headers.set("Transfer-Encoding", "chunked");
		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			if(it->second.empty()){
				continue;
			}
			data.put(it->first.get());
			data.put(": ");
			data.put(it->second.data(), it->second.size());
			data.put("\r\n");
		}
		data.put("\r\n");

		m_session->send(STD_MOVE(data));

		m_session.reset();
	}
}

}
