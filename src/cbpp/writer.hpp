#ifndef POSEIDON_CBPP_WRITER_HPP_
#define POSEIDON_CBPP_WRITER_HPP_

#include <string>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "control_codes.hpp"

namespace Poseidon {

namespace Cbpp {
	class Writer {
	public:
		Writer();
		virtual ~Writer();

	protected:
		virtual long onEncodedDataAvail(StreamBuffer encoded) = 0;

	public:
		long putDataMessage(boost::uint16_t messageId, StreamBuffer payload);

		long putControlMessage(ControlCode controlCode, boost::int64_t vintParam, std::string stringParam);
	};
}

}

#endif
