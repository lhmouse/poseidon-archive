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
#include "../tcp_session_base.hpp"
#include "../optional_map.hpp"
#include "verbs.hpp"
#include "status_codes.hpp"

namespace Poseidon {

namespace Http {
	class UpgradedSessionBase;

	class Session : public TcpSessionBase {
		friend UpgradedSessionBase;

	private:
		enum State {
			S_FIRST_HEADER,
			S_HEADERS,
			S_CONTENTS,
		};

	private:
		class RequestJob;
		class ErrorJob;

		class HeaderParser;

	public:
		typedef const std::vector<std::string> BasicAuthInfo;

	private:
		boost::shared_ptr<BasicAuthInfo> m_authInfo;

		State m_state;
		boost::uint64_t m_totalLength;
		boost::uint64_t m_contentLength;
		std::string m_line;
		bool m_chunked;
		std::string m_chunk;

		Verb m_verb;
		std::string m_uri;
		unsigned m_version;	// x * 10000 + y 表示 HTTP x.y
		OptionalMap m_getParams;
		OptionalMap m_headers;

		mutable boost::mutex m_upgreadedMutex;
		boost::shared_ptr<UpgradedSessionBase> m_upgradedSession;

	public:
		explicit Session(UniqueFile socket,
			boost::shared_ptr<BasicAuthInfo> authInfo = VAL_INIT // 必须是排序的，否则会抛出异常。
			);
		~Session();

	private:
		void onReadAvail(const void *data, std::size_t size) OVERRIDE FINAL;

	protected:
		virtual void onRequest(Verb verb, std::string uri, unsigned version,
			OptionalMap getParams, OptionalMap headers, std::string contents) = 0;

		virtual boost::shared_ptr<UpgradedSessionBase> onUpgrade(const std::string &type,
			Verb verb, const std::string &uri, unsigned version, const OptionalMap &params, const OptionalMap &headers);

	public:
		boost::shared_ptr<UpgradedSessionBase> getUpgradedSession() const;

		bool send(StatusCode statusCode, OptionalMap headers, StreamBuffer contents, bool fin = false);
		bool send(StatusCode statusCode, StreamBuffer contents = StreamBuffer(), bool fin = false){
			return send(statusCode, OptionalMap(), STD_MOVE(contents), fin);
		}

		bool sendDefault(StatusCode statusCode, OptionalMap headers, bool fin = false);
		bool sendDefault(StatusCode statusCode, bool fin = false){
			return sendDefault(statusCode, OptionalMap(), fin);
		}
	};
}

}

#endif
