#include "../precompiled.hpp"
#include "chunked_writer.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace Http {
	ChunkedWriter::ChunkedWriter(){
	}
	ChunkedWriter::ChunkedWriter(boost::shared_ptr<Session> session, StatusCode statusCode, OptionalMap headers){
		reset(STD_MOVE(session), statusCode, STD_MOVE(headers));
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
	void ChunkedWriter::reset(boost::shared_ptr<Session> session, StatusCode statusCode, OptionalMap headers){
		PROFILE_ME;

		if(m_session != session){
			reset();
		}

		if(!session){
			LOG_POSEIDON_ERROR("No session specified.");
			DEBUG_THROW(Exception, ST_INTERNAL_SERVER_ERROR);
		}

		StreamBuffer data;

		char first[64];
		unsigned len = (unsigned)std::sprintf(first, "HTTP/1.1 %u ", static_cast<unsigned>(statusCode));
		data.put(first, len);
		const AUTO(desc, getStatusCodeDesc(statusCode));
		data.put(desc.descShort);
		data.put("\r\n");

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

		session->TcpSessionBase::send(STD_MOVE(data), false);

		m_session = STD_MOVE(session); // noexcept
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

		m_session->TcpSessionBase::send(STD_MOVE(data), false);
	}

	void ChunkedWriter::finalize(OptionalMap headers, bool fin){
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

		m_session->TcpSessionBase::send(STD_MOVE(data), fin);

		m_session.reset();
	}
}

}
