// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_AUTHORIZATION_HPP_
#define POSEIDON_HTTP_AUTHORIZATION_HPP_

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "verbs.hpp"
#include "../ip_port.hpp"
#include "../optional_map.hpp"

namespace Poseidon {

namespace Http {
	class Header;

	enum AuthResult {
		AUTH_SUCCEEDED				= 0,
		AUTH_REQUIRED				= 1,
		AUTH_INVALID_HEADER			= 2,
		AUTH_UNKNOWN_SCHEME			= 3,
		AUTH_INVALID_USER_PASS		= 4,
		AUTH_INACCEPTABLE_NONCE		= 5,
		AUTH_EXPIRED				= 6,
		AUTH_INACCEPTABLE_ALGORITHM	= 7,
		AUTH_INACCEPTABLE_QOP		= 8,
	};

	class AuthInfo; // 没有定义的类，当作句柄使用。

	extern boost::shared_ptr<const AuthInfo> createAuthInfo(std::vector<std::string> basicUserPass); // username:password

	// 支持 Basic 和 Digest。如果返回值的 first 成员为 AUTH_SUCCEEDED，second 为指向认证成功的 username:password 的指针。
	extern std::pair<AuthResult, const std::string *> checkAuthorizationHeader(
		const boost::shared_ptr<const AuthInfo> &authInfo, const IpPort &remoteAddr, Verb verb, const std::string &authHeader);
	extern void throwUnauthorized(AuthResult authResult, const IpPort &remoteAddr,
		bool isProxy = false, OptionalMap headers = VAL_INIT) __attribute__((__noreturn__));

	// 如果 authInfo 为空指针，返回空指针；否则，返回指向认证成功的 username:password 的指针。
	extern const std::string *checkAndThrowIfUnauthorized(
		const boost::shared_ptr<const AuthInfo> &authInfo, const IpPort &remoteAddr, const Header &header,
		bool isProxy = false, OptionalMap responseHeaders = VAL_INIT);
}

}

#endif
