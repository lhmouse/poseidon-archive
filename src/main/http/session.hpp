// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include "../cxx_ver.hpp"
#include <string>
#include <set>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include "../tcp_session_base.hpp"
#include "../optional_map.hpp"
#include "verb.hpp"
#include "status.hpp"

namespace Poseidon {

class HttpServer;
class HttpUpgradedSessionBase;

class HttpSession : public TcpSessionBase {
	friend HttpServer;
	friend HttpUpgradedSessionBase;

private:
	enum State {
		ST_FIRST_HEADER,
		ST_HEADERS,
		ST_CONTENTS,
	};

private:
	const std::size_t m_category;

	State m_state;
	boost::uint64_t m_totalLength;
	boost::uint64_t m_contentLength;
	std::string m_line;

	boost::shared_ptr<HttpUpgradedSessionBase> m_upgradedSession;
	boost::shared_ptr<std::set<std::string> > m_authInfo;

	HttpVerb m_verb;
	unsigned m_version;	// x * 10000 + y 表示 HTTP x.y
	std::string m_uri;
	OptionalMap m_getParams;
	OptionalMap m_headers;

public:
	HttpSession(std::size_t category, UniqueFile socket);
	~HttpSession();

private:
	void onReadAvail(const void *data, std::size_t size) OVERRIDE FINAL;

	void onAllHeadersRead();

public:
	std::size_t getCategory() const {
		return m_category;
	}

	void setAuthInfo(boost::shared_ptr<std::set<std::string> > authInfo){
		m_authInfo.swap(authInfo);
	}

	bool send(HttpStatus status, OptionalMap headers, StreamBuffer contents, bool fin = false);
	bool send(HttpStatus status, StreamBuffer contents = StreamBuffer(), bool fin = false){
		return send(status, OptionalMap(), STD_MOVE(contents), fin);
	}

	bool sendDefault(HttpStatus status, OptionalMap headers, bool fin = false);
	bool sendDefault(HttpStatus status, bool fin = false){
		return sendDefault(status, OptionalMap(), fin);
	}
};

}

#endif
