// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CLIENT_HPP_
#define POSEIDON_HTTP_CLIENT_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include "header.hpp"
#include "verbs.hpp"
#include "../tcp_client_base.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace Http {
	class Client : public TcpClientBase {
	private:
		class HeaderJob;
		class RequestJob;

	private:
		enum State {
			S_FIRST_HEADER		= 0,
			S_HEADERS			= 1,
			S_END_OF_ENTITY		= 2,
			S_IDENTITY			= 3,
			S_CHUNK_HEADER		= 4,
			S_CHUNK_DATA		= 5,
			S_CHUNKED_TRAILER	= 6,
		};

	protected:
		enum {
			CONTENT_CHUNKED = (boost::uint64_t)-1,
		};

	private:
		StreamBuffer m_received;

		bool m_expectingNewLine;
		boost::uint64_t m_sizeExpecting;
		State m_state;

		Header m_header;
		StreamBuffer m_entity;

	public:
		explicit Client(const IpPort &addr, boost::uint64_t keepAliveTimeout, bool useSsl);
		~Client();

	private:
		void onReadAvail(const void *data, std::size_t size) FINAL;
		void onReadHup() NOEXCEPT OVERRIDE;

	protected:
		// 和 Http::Session 不同，这个函数在主线程中调用。
		// 如果 Transfer-Encoding 是 chunked， contentLength 的值为 CONTENT_CHUNKED。
		virtual void onHeader(const Header &header, boost::uint64_t contentLength);

		// 报文可能分几次收到。
		virtual void onResponse(boost::uint64_t contentLength, boost::uint64_t contentOffset, const StreamBuffer &entity) = 0;

	public:
		bool send(Verb verb, const std::string &uri, OptionalMap headers = VAL_INIT, StreamBuffer entity = VAL_INIT, bool fin = false);
		bool send(Verb verb, const std::string &uri, StreamBuffer entity, bool fin = false){
			return send(verb, uri, OptionalMap(), STD_MOVE(entity), fin);
		}
	};
}

}

#endif
