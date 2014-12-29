#ifndef POSEIDON_PLAYER_CLIENT_HPP_
#define POSEIDON_PLAYER_CLIENT_HPP_

#include "../tcp_client_base.hpp"
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "protocol_base.hpp"
#include "status.hpp"

namespace Poseidon {

class TimerItem;

class PlayerClient : public TcpClientBase {
private:
	boost::shared_ptr<const TimerItem> m_keepAliveTimer;

	boost::uint64_t m_payloadLen;
	unsigned m_protocolId;
	StreamBuffer m_payload;

protected:
	explicit PlayerClient(const IpPort &addr, unsigned long long keepAliveTimeout, bool useSsl);
	~PlayerClient();

private:
	void onReadAvail(const void *data, std::size_t size) OVERRIDE FINAL;

public:
	virtual void onResponse(boost::uint16_t protocolId, StreamBuffer contents) = 0;
	virtual void onError(boost::uint16_t protocolId, PlayerStatus status, std::string reason) = 0;

public:
	bool send(boost::uint16_t protocolId, StreamBuffer contents, bool fin = false);

	template<class ProtocolT>
	typename boost::enable_if<boost::is_base_of<ProtocolBase, ProtocolT>, bool>::type
		send(const ProtocolT &contents, bool fin = false)
	{
		return send(ProtocolT::ID, StreamBuffer(contents), fin);
	}
};

}

#endif
