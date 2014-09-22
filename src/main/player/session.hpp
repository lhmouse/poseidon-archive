#ifndef POSEIDON_PLAYER_SESSION_HPP_
#define POSEIDON_PLAYER_SESSION_HPP_

#include "../../cxx_ver.hpp"
#include <cstddef>
#include <boost/cstdint.hpp>
#include "../tcp_session_base.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

class PlayerSession : public TcpSessionBase {
private:
	int m_payloadLen;
	unsigned m_protocolId;
	StreamBuffer m_payload;

public:
	explicit PlayerSession(Move<ScopedFile> socket);
	~PlayerSession();

protected:
	void onReadAvail(const void *data, std::size_t size);

public:
	// 这两个函数是线程安全的。
	void send(boost::uint16_t status, const StreamBuffer &payload);
	void sendUsingMove(boost::uint16_t status, StreamBuffer &payload);
#ifdef POSEIDON_CXX11
	void send(boost::uint16_t status, StreamBuffer &&payload){
		sendUsingMove(status, payload);
	}
	void sendUsingMove(boost::uint16_t status, StreamBuffer &&payload){
		sendUsingMove(status, payload);
	}
#endif
};

}

#endif
