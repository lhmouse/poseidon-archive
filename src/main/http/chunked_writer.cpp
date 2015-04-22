#include "../precompiled.hpp"
#include "chunked_writer.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../optional_map.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace Http {
	ChunkedWriter::ChunkedWriter(boost::shared_ptr<Session> session)
		: m_session(STD_MOVE(session)), m_inProgress(false)
	{
	}
	ChunkedWriter::~ChunkedWriter(){
		reset();
	}

	void ChunkedWriter::reset(boost::shared_ptr<Session> session) NOEXCEPT {
		PROFILE_ME;

		if(m_session == session){
			return;
		}

		if(m_inProgress){
			LOG_POSEIDON_DEBUG("ChunkedWriter is not finalized, shutting down session...");
			m_session->forceShutdown();
		}

		m_session = STD_MOVE(session);
		m_inProgress = false;
	}

	void ChunkedWriter::initialize(StatusCode statusCode, OptionalMap headers){
		if(!m_session){
			LOG_POSEIDON_ERROR("No session specified.");
			DEBUG_THROW(Exception, ST_INTERNAL_SERVER_ERROR);
		}
		if(m_inProgress){
			LOG_POSEIDON_ERROR("Chunked writer has already been initialized.");
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

		m_session->TcpSessionBase::send(STD_MOVE(data), false);

		m_inProgress = true;
	}

	void ChunkedWriter::put(const void *data, std::size_t size){
		put(StreamBuffer(data, size));
	}
	void ChunkedWriter::put(const char *str){
		put(StreamBuffer(str));
	}
	void ChunkedWriter::put(StreamBuffer buffer){
		PROFILE_ME;

		if(!m_inProgress){
			LOG_POSEIDON_ERROR("Chunked writer has not been initialized.");
			DEBUG_THROW(Exception, ST_INTERNAL_SERVER_ERROR);
		}

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
		if(!m_inProgress){
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

		m_inProgress = false;
	}
	void ChunkedWriter::finalize(bool fin){
		finalize(OptionalMap(), fin);
	}
}

}
