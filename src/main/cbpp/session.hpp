// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_SESSION_HPP_
#define POSEIDON_CBPP_SESSION_HPP_

#include <cstddef>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/cstdint.hpp>
#include "../tcp_session_base.hpp"
#include "../stream_buffer.hpp"
#include "status.hpp"

namespace Poseidon {

class CbppMessageBase;

class CbppSession : public TcpSessionBase {
	friend class CbppServer;

private:
	const std::size_t m_category;

	boost::uint64_t m_payloadLen;
	unsigned m_messageId;
	StreamBuffer m_payload;

public:
	CbppSession(std::size_t category, UniqueFile socket);
	~CbppSession();

private:
	void onReadAvail(const void *data, std::size_t size) OVERRIDE FINAL;

public:
	std::size_t getCategory() const {
		return m_category;
	}

	bool send(boost::uint16_t messageId, StreamBuffer contents, bool fin = false);

	template<class MessageT>
	typename boost::enable_if<boost::is_base_of<CbppMessageBase, MessageT>, bool>::type
		send(const MessageT &contents, bool fin = false)
	{
		return send(MessageT::ID, StreamBuffer(contents), fin);
	}

	bool sendError(boost::uint16_t messageId, CbppStatus status, std::string reason, bool fin = false);
	bool sendError(boost::uint16_t messageId, CbppStatus status, bool fin = false){
		return sendError(messageId, status, std::string(), fin);
	}
};

}

#endif
