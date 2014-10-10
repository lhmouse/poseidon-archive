#ifndef POSEIDON_PLAYER_SESSION_HPP_
#define POSEIDON_PLAYER_SESSION_HPP_

#include <cstddef>
#include <boost/cstdint.hpp>
#include "../tcp_session_base.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

class PlayerSession : public TcpSessionBase {
	friend class PlayerServer;

private:
	int m_payloadLen;
	unsigned m_protocolId;
	StreamBuffer m_incoming;

public:
	explicit PlayerSession(Move<ScopedFile> socket);
	~PlayerSession();

public:
	void onReadAvail(const void *data, std::size_t size);
	bool send(boost::uint16_t status, StreamBuffer payload);
};

}

#endif
