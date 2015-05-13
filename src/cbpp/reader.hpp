#ifndef POSEIDON_CBPP_READER_HPP_
#define POSEIDON_CBPP_READER_HPP_

#include <string>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "control_codes.hpp"

namespace Poseidon {

namespace Cbpp {
	class Reader {
	private:
		enum State {
			S_PAYLOAD_SIZE		= 0,
			S_EX_PAYLOAD_SIZE	= 1,
			S_MESSAGE_ID		= 2,
			S_PAYLOAD			= 3,
		};

	private:
		StreamBuffer m_internal;

		boost::uint64_t m_sizeExpecting;
		State m_state;

		boost::uint64_t m_payloadSize;
		boost::uint16_t m_messageId;
		boost::uint64_t m_payloadOffset;

	public:
		Reader();
		virtual ~Reader();

	protected:
		virtual void onDataMessageHeader(boost::uint16_t messageId, boost::uint64_t payloadSize) = 0;
		virtual void onDataMessagePayload(boost::uint64_t payloadOffset, StreamBuffer payload) = 0;
		virtual void onDataMessageEnd(boost::uint64_t payloadSize) = 0;

		virtual void onControlMessage(ControlCode controlCode, boost::int64_t vintParam, std::string stringParam) = 0;

	public:
		void putEncodedData(StreamBuffer encoded);
	};
}

}

#endif
