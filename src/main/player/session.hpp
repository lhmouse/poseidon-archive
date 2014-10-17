#ifndef POSEIDON_PLAYER_SESSION_HPP_
#define POSEIDON_PLAYER_SESSION_HPP_

#include <cstddef>
#include <boost/cstdint.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>
#include "../tcp_session_base.hpp"
#include "../stream_buffer.hpp"
#include "status.hpp"

namespace Poseidon {

class ProtocolBase;

class PlayerSession : public TcpSessionBase {
	friend class PlayerServer;

private:
	int m_payloadLen;
	unsigned m_protocolId;
	StreamBuffer m_payload;

public:
	explicit PlayerSession(Move<ScopedFile> socket);
	~PlayerSession();

public:
	void onReadAvail(const void *data, std::size_t size);

	bool shutdown(PlayerStatus status,
		boost::uint16_t protocolId = 0, StreamBuffer additional = StreamBuffer());

	bool send(boost::uint16_t protocolId, StreamBuffer protocolData);

	template<class ProtocolT>
	typename boost::enable_if<boost::is_base_of<ProtocolBase, ProtocolT>, bool>::type
		send(boost::uint16_t protocolId, const ProtocolT &protocolData)
	{
		StreamBuffer temp;
		protocolData >> temp;
		return send(protocolId, STD_MOVE(temp));
	}
};

}

#endif
