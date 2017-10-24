// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "authentication.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../buffer_streams.hpp"
#include "../base64.hpp"
#include "../time.hpp"
#include "../random.hpp"
#include "../md5.hpp"
#include "exception.hpp"
#include <openssl/aes.h>

namespace Poseidon {
namespace Http {

class AuthenticationContext : NONCOPYABLE {
private:
	struct PasswordComparator {
		bool operator()(const std::string &lhs, const char *rhs) const {
			return ::strcasecmp(lhs.c_str(), rhs);
		}
		bool operator()(const char *lhs, const std::string &rhs) const {
			return ::strcasecmp(lhs, rhs.c_str());
		}
	};

private:
	const std::string m_realm;

	std::vector<std::string> m_passwords;

public:
	explicit AuthenticationContext(std::string realm)
		: m_realm(STD_MOVE(realm))
	{ }

public:
	const std::string &get_realm() const {
		return m_realm;
	}

	std::pair<const char *, const char *> get_password(const char *username) const {
		const AUTO(range, std::equal_range(m_passwords.begin(), m_passwords.end(), username, PasswordComparator()));
		if(range.first == range.second){
			return VAL_INIT;
		}
		return std::make_pair(range.first->c_str(), range.first->c_str() + std::strlen(username) + 1);
	}
	void set_password(const char *username, const char *password){
		std::string str;
		str += username;
		str += '\0';
		str += password;
		const AUTO(range, std::equal_range(m_passwords.begin(), m_passwords.end(), username, PasswordComparator()));
		if(range.first == range.second){
			m_passwords.insert(range.first, STD_MOVE(str));
		} else {
			range.first->swap(str);
		}
	}
};

boost::shared_ptr<const AuthenticationContext> create_authentication_context(
	const std::string &realm, const std::vector<std::string> &basic_user_pass)
{
	PROFILE_ME;

	if(basic_user_pass.empty()){
		LOG_POSEIDON_ERROR("No username:password provided!");
		DEBUG_THROW(BasicException, sslit("No username:password provided"));
	}

	AUTO(context, boost::make_shared<AuthenticationContext>(realm));
	std::string str;
	for(AUTO(it, basic_user_pass.begin()); it != basic_user_pass.end(); ++it){
		str = *it;
		AUTO(pos, str.find('\0'));
		if(pos != std::string::npos){
			LOG_POSEIDON_ERROR("Username or password shall not contain null characters: ", str);
			DEBUG_THROW(BasicException, sslit("Username or password shall not contain null characters"));
		}
		pos = str.find(':');
		if(pos == std::string::npos){
			LOG_POSEIDON_ERROR("Colon delimiter not found: ", str);
			DEBUG_THROW(BasicException, sslit("Colon delimiter not found"));
		}
		str.at(pos) = '\0';
		const AUTO(old_password, context->get_password(str.c_str()).second);
		if(old_password){
			LOG_POSEIDON_ERROR("Duplicate username: ", str);
			DEBUG_THROW(BasicException, sslit("Duplicate username"));
		}
		context->set_password(str.c_str(), str.c_str() + pos + 1);
	}
	return STD_MOVE_IDN(context);
}
std::pair<AuthenticationResult, const char *> check_authentication(
	const boost::shared_ptr<const AuthenticationContext> &context, bool is_proxy, const IpPort &remote_addr, const RequestHeaders &request_headers)
{
	PROFILE_ME;

	if(!context){
		LOG_POSEIDON_INFO("HTTP authentication succeeded (assuming anonymous).");
		return std::make_pair(AUTH_SUCCEEDED, NULLPTR);
	}
	const AUTO_REF(header_value, request_headers.headers.get(is_proxy ? "Proxy-Authorization" : "Authorization"));
	if(header_value.empty()){
		return std::make_pair(AUTH_HEADER_NOT_SET, NULLPTR);
	}
	if(::strncasecmp(header_value.c_str(), "Basic ", 6) == 0){
		return check_authentication_basic(context, header_value);
	}
	if(::strncasecmp(header_value.c_str(), "Digest ", 7) == 0){
		return check_authentication_digest(context, remote_addr, request_headers.verb, header_value);
	}
	LOG_POSEIDON_WARNING("HTTP authentication scheme not supported: ", header_value);
	return std::make_pair(AUTH_SCHEME_NOT_SUPPORTED, NULLPTR);
}
__attribute__((__noreturn__)) void throw_authentication_failure(
	const std::string &realm, bool is_proxy, const IpPort &remote_addr, AuthenticationResult result)
{
	PROFILE_ME;

	throw_authentication_failure_digest(realm, is_proxy, remote_addr, result);
}
const char *check_authentication_simple(
	const boost::shared_ptr<const AuthenticationContext> &context, bool is_proxy, const IpPort &remote_addr, const RequestHeaders &request_headers)
{
	PROFILE_ME;

	const AUTO(pair, check_authentication(context, is_proxy, remote_addr, request_headers));
	if(pair.first != AUTH_SUCCEEDED){
		DEBUG_THROW_ASSERT(context);
		throw_authentication_failure(context->get_realm(), is_proxy, remote_addr, pair.first);
	}
	return pair.second;
}

namespace {
	__attribute__((__noreturn__)) void do_throw_authentication_failure(bool is_proxy, std::string auth_header){
		PROFILE_ME;

		Poseidon::OptionalMap headers;
		headers.set(SharedNts::view(is_proxy ? "Proxy-Authenticate" : "WWW-Authenticate"), STD_MOVE(auth_header));
		DEBUG_THROW(Exception, is_proxy ? ST_PROXY_AUTH_REQUIRED : ST_UNAUTHORIZED, STD_MOVE(headers));
	}
}

// Basic
std::pair<AuthenticationResult, const char *> check_authentication_basic(
	const boost::shared_ptr<const AuthenticationContext> &context, const std::string &header_value)
{
	PROFILE_ME;

	if(!context){
		LOG_POSEIDON_INFO("HTTP authentication succeeded (assuming anonymous).");
		return std::make_pair(AUTH_SUCCEEDED, NULLPTR);
	}
	if(::strncasecmp(header_value.c_str(), "Basic ", 6) != 0){
		LOG_POSEIDON_WARNING("HTTP authentication scheme not supported: ", header_value);
		return std::make_pair(AUTH_SCHEME_NOT_SUPPORTED, NULLPTR);
	}
	Base64Decoder base64_dec;
	base64_dec.put(header_value.c_str() + 7);
	AUTO(buffer, base64_dec.finalize());
	const AUTO(auth_str, static_cast<char *>(buffer.squash()));
	const AUTO(colon, std::strchr(auth_str, ':'));
	if(!colon){
		LOG_POSEIDON_ERROR("Colon delimiter not found: ", auth_str);
		return std::make_pair(AUTH_HEADER_FORMAT_ERROR, NULLPTR);
	}
	*colon = 0;
	const AUTO(pair, context->get_password(auth_str));
	if(!pair.second){
		LOG_POSEIDON_DEBUG("User not found: ", auth_str);
		return std::make_pair(AUTH_PASSWORD_INCORRECT, NULLPTR);
	}
	if(std::strcmp(colon + 1, pair.second) != 0){
		LOG_POSEIDON_DEBUG("Password incorrect: ", colon + 1);
		return std::make_pair(AUTH_PASSWORD_INCORRECT, NULLPTR);
	}
	LOG_POSEIDON_INFO("HTTP authentication succeeded (using password via the Basic scheme): ", pair.first);
	return std::make_pair(AUTH_SUCCEEDED, pair.first);

}
__attribute__((__noreturn__)) void throw_authentication_failure_basic(
	const std::string &realm, bool is_proxy, AuthenticationResult result)
{
	PROFILE_ME;

	Buffer_ostream os;
	os <<"Basic realm=\"" <<realm <<"\""
	   <<", stale=" <<std::boolalpha <<(result == AUTH_REQUEST_EXPIRED);
	do_throw_authentication_failure(is_proxy, os.get_buffer().dump_string());
}
/*
// Digest
std::pair<AuthenticationResult, const char *> check_authentication_digest(
	const boost::shared_ptr<const AuthenticationContext> &context, const IpPort &remote_addr, Verb verb, const std::string &header_value)
{
	PROFILE_ME;

}*/
__attribute__((__noreturn__)) void throw_authentication_failure_digest(
	const std::string &realm, bool is_proxy, const IpPort &remote_addr, AuthenticationResult result)
{
	PROFILE_ME;

	DEBUG_THROW_ASSERT(result != AUTH_SUCCEEDED);

	Buffer_ostream os;
	os <<"Digest realm=\"" <<realm <<"\""
	   <<", stale=" <<std::boolalpha <<(result == AUTH_REQUEST_EXPIRED)
	   <<", algorithm=\"MD5\", qop-options=\"auth\""
	   <<", nonce=\"""\"";
	do_throw_authentication_failure(is_proxy, os.get_buffer().dump_string());
}

}
}
