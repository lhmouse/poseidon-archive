#ifndef POSEIDON_PLAYER_SESSION_HPP_
#define POSEIDON_PLAYER_SESSION_HPP_

#include <cstddef>
#include <boost/cstdint.hpp>
#include "tcp_session_base.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

class PlayerSession : public TcpSessionBase {
private:
	int m_payloadLen;
	unsigned m_protocolId;
	StreamBuffer m_payload;

public:
	explicit PlayerSession(ScopedFile &socket);

protected:
	void onReadAvail(const void *data, std::size_t size);
	void onReadComplete();

public:
	void send(boost::uint16_t status, const StreamBuffer &payload);
	void sendWithMove(boost::uint16_t status, StreamBuffer &payload);
};

}

#endif
