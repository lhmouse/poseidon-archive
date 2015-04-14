// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include "../tcp_session_base.hpp"
#include "../optional_map.hpp"
#include "verbs.hpp"
#include "status_codes.hpp"

namespace Poseidon {

namespace Http {
	class Server;
	class UpgradedSessionBase;

	class Session : public TcpSessionBase {
		friend Server;
		friend UpgradedSessionBase;

	private:
		enum State {
			S_FIRST_HEADER,
			S_HEADERS,
			S_CONTENTS,
		};

	private:
		class HeaderParser;

	private:
		const std::size_t m_category;

		State m_state;
		boost::uint64_t m_totalLength;
		boost::uint64_t m_contentLength;
		std::string m_line;

		boost::shared_ptr<UpgradedSessionBase> m_upgradedSession;
		boost::shared_ptr<const std::vector<std::string> > m_authInfo;

		Verb m_verb;
		unsigned m_version;	// x * 10000 + y 表示 HTTP x.y
		std::string m_uri;
		OptionalMap m_getParams;
		OptionalMap m_headers;

	public:
		Session(std::size_t category, UniqueFile socket);
		~Session();

	private:
		void onReadAvail(const void *data, std::size_t size) OVERRIDE FINAL;

	public:
		std::size_t getCategory() const {
			return m_category;
		}

		void setAuthInfo(boost::shared_ptr<const std::vector<std::string> > authInfo){
			m_authInfo = STD_MOVE(authInfo);
		}

		const boost::shared_ptr<UpgradedSessionBase> &getUpgradedSession() const {
			return m_upgradedSession;
		}

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
