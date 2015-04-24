// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/cstdint.hpp>
#include "header.hpp"
#include "status_codes.hpp"
#include "../tcp_session_base.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace Http {
	class UpgradedSessionBase;

	class Session : public TcpSessionBase {
		friend UpgradedSessionBase;

	private:
		enum State {
			S_FIRST_HEADER		= 0,
			S_HEADERS			= 1,
			S_UPGRADED			= 2,
			S_END_OF_ENTITY		= 3,
			S_IDENTITY			= 4,
			S_CHUNK_HEADER		= 5,
			S_CHUNK_DATA		= 6,
			S_CHUNKED_TRAILER	= 7,
		};

	private:
		class RequestJob;
		class ErrorJob;

		class UpgradeJob;

	public:
		typedef const std::vector<std::string> BasicAuthInfo;

	private:
		const boost::shared_ptr<BasicAuthInfo> m_authInfo;

		mutable boost::mutex m_upgreadedMutex;
		boost::shared_ptr<UpgradedSessionBase> m_upgradedSession;

		StreamBuffer m_received;

		boost::uint64_t m_sizeTotal;
		bool m_expectingNewLine;
		boost::uint64_t m_sizeExpecting;
		State m_state;

		Header m_header;
		StreamBuffer m_entity;

	public:
		explicit Session(UniqueFile socket,
			boost::shared_ptr<BasicAuthInfo> authInfo = VAL_INIT // 必须是排序的，否则会抛出异常。
			);
		~Session();

	private:
		void onReadAvail(const void *data, std::size_t size) FINAL;
		void onReadHup() NOEXCEPT OVERRIDE;

	protected:
		virtual void onRequest(const Header &header, const StreamBuffer &entity) = 0;

		virtual boost::shared_ptr<UpgradedSessionBase> onUpgrade(const std::string &type,
			const Header &header, const StreamBuffer &entity);

	public:
		boost::shared_ptr<UpgradedSessionBase> getUpgradedSession() const;

		bool send(StatusCode statusCode, OptionalMap headers, StreamBuffer entity, bool fin = false);
		bool send(StatusCode statusCode, StreamBuffer entity = StreamBuffer(), bool fin = false){
			return send(statusCode, OptionalMap(), STD_MOVE(entity), fin);
		}

		bool sendDefault(StatusCode statusCode, OptionalMap headers, bool fin = false);
		bool sendDefault(StatusCode statusCode, bool fin = false){
			return sendDefault(statusCode, OptionalMap(), fin);
		}
	};
}

}

#endif
