// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "authentication.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../buffer_streams.hpp"
#include "../base64.hpp"
#include "../string.hpp"
#include "../time.hpp"
#include "../random.hpp"
#include "../md5.hpp"
#include "../checked_arithmetic.hpp"
#include "../singletons/main_config.hpp"
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
	const boost::shared_ptr<const AuthenticationContext> &context, bool is_proxy, const IpPort &remote_info, const RequestHeaders &request_headers)
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
		return check_authentication_digest(context, remote_info, request_headers.verb, header_value);
	}
	LOG_POSEIDON_WARNING("HTTP authentication scheme not supported: ", header_value);
	return std::make_pair(AUTH_SCHEME_NOT_SUPPORTED, NULLPTR);
}
__attribute__((__noreturn__)) void throw_authentication_failure(
	const boost::shared_ptr<const AuthenticationContext> &context, bool is_proxy, const IpPort &remote_info, AuthenticationResult result)
{
	PROFILE_ME;

	throw_authentication_failure_digest(context, is_proxy, remote_info, result);
}
const char *check_authentication_simple(
	const boost::shared_ptr<const AuthenticationContext> &context, bool is_proxy, const IpPort &remote_info, const RequestHeaders &request_headers)
{
	PROFILE_ME;

	const AUTO(pair, check_authentication(context, is_proxy, remote_info, request_headers));
	if(pair.first != AUTH_SUCCEEDED){
		DEBUG_THROW_ASSERT(context);
		throw_authentication_failure(context, is_proxy, remote_info, pair.first);
	}
	return pair.second;
}

namespace {
	class StringQuoter {
	private:
		const char *m_str;

	public:
		StringQuoter(const char *str)
			: m_str(str)
		{ }

	public:
		const char *get() const {
			return m_str;
		}
	};

	std::ostream &operator<<(std::ostream &os, const StringQuoter &rhs){
		PROFILE_ME;

		const char *read = rhs.get();
		int ch;
		os <<'\"';
		while((ch = static_cast<unsigned char>(*(read++))) != 0){
			if((ch == '\"') || (ch == '\\')){
				os <<'\\';
			}
			os <<static_cast<char>(ch);
		}
		os <<'\"';
		return os;
	}

	__attribute__((__noreturn__)) void do_throw_authentication_failure(bool is_proxy, std::string authenticate_str){
		PROFILE_ME;

		Poseidon::OptionalMap headers;
		headers.set(SharedNts::view(is_proxy ? "Proxy-Authenticate" : "WWW-Authenticate"), STD_MOVE(authenticate_str));
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
	if(header_value.empty()){
		return std::make_pair(AUTH_HEADER_NOT_SET, NULLPTR);
	}
	if(::strncasecmp(header_value.c_str(), "Basic ", 6) != 0){
		LOG_POSEIDON_WARNING("HTTP authentication scheme not supported: ", header_value);
		return std::make_pair(AUTH_SCHEME_NOT_SUPPORTED, NULLPTR);
	}

	AUTO(auth_str, base64_decode(header_value.c_str() + 6));
	const AUTO(colon_pos, auth_str.find(':'));
	if(colon_pos == std::string::npos){
		LOG_POSEIDON_ERROR("Colon delimiter not found: ", auth_str);
		return std::make_pair(AUTH_HEADER_FORMAT_ERROR, NULLPTR);
	}
	auth_str.at(colon_pos) = 0;

	const AUTO(pair, context->get_password(auth_str.c_str()));
	if(!pair.second){
		LOG_POSEIDON_DEBUG("User not found: ", auth_str);
		return std::make_pair(AUTH_PASSWORD_INCORRECT, NULLPTR);
	}
	if(::strcmp(auth_str.c_str() + colon_pos + 1, pair.second) != 0){
		LOG_POSEIDON_DEBUG("Password incorrect: ", auth_str);
		return std::make_pair(AUTH_PASSWORD_INCORRECT, NULLPTR);
	}
	LOG_POSEIDON_INFO("HTTP authentication succeeded (using password via the Basic scheme): ", pair.first);
	return std::make_pair(AUTH_SUCCEEDED, pair.first);
}
__attribute__((__noreturn__)) void throw_authentication_failure_basic(
	const boost::shared_ptr<const AuthenticationContext> &context, bool is_proxy, AuthenticationResult result)
{
	PROFILE_ME;
	DEBUG_THROW_ASSERT(result != AUTH_SUCCEEDED);

	Buffer_ostream bos;
	bos <<"Basic realm=" <<StringQuoter(context->get_realm().c_str());

	do_throw_authentication_failure(is_proxy, bos.get_buffer().dump_string());
}

namespace {
	const boost::uint32_t g_server_id = random_uint32();

	struct Nonce {
		boost::uint32_t server_id; // g_server_id
		boost::uint32_t reserved;
		boost::uint64_t timestamp; // get_utc_time()
	};
	BOOST_STATIC_ASSERT(sizeof(Nonce) == 16);

	void create_nonce(Nonce *nonce){
		nonce->server_id = g_server_id;
		nonce->timestamp = get_utc_time();
		nonce->reserved  = 0;
	}
	void encrypt_nonce(char *str, const Nonce *nonce, const char *key){
		Md5_ostream md5_os;
		md5_os <<key;
		AUTO(md5, md5_os.finalize());
		::AES_KEY aes_key;
		int err_code = ::AES_set_encrypt_key(md5.data(), 128, &aes_key);
		DEBUG_THROW_ASSERT(err_code == 0);
		boost::array<unsigned char, 16> out;
		::AES_encrypt(reinterpret_cast<const unsigned char *>(nonce), out.data(), &aes_key);
		char *write = str;
		for(unsigned i = 0; i < 16; ++i){
			const unsigned hi = out[i] / 16;
			const unsigned lo = out[i] % 16;
			*(write++) = 'a' + hi;
			*(write++) = 'a' + lo;
		}
		*write = 0;
	}
	bool decrypt_nonce(Nonce *nonce, const char *str, const char *key){
		Md5_ostream md5_os;
		md5_os <<key;
		AUTO(md5, md5_os.finalize());
		::AES_KEY aes_key;
		int err_code = ::AES_set_decrypt_key(md5.data(), 128, &aes_key);
		DEBUG_THROW_ASSERT(err_code == 0);
		boost::array<unsigned char, 16> in;
		const char *read = str;
		for(unsigned i = 0; i < 16; ++i){
			const unsigned hi = static_cast<unsigned>(*(read++)) - 'a';
			if(hi >= 16){
				return false;
			}
			const unsigned lo = static_cast<unsigned>(*(read++)) - 'a';
			if(lo >= 16){
				return false;
			}
			in[i] = (hi << 4) | lo;
		}
		if(*read != 0){
			return false;
		}
		::AES_decrypt(in.data(), reinterpret_cast<unsigned char *>(nonce), &aes_key);
		return true;
	}

	void finalize_as_hex(char *str, Md5_ostream &md5_os){
		static CONSTEXPR const char HEX_TABLE[] = "0123456789abcdef";
		const AUTO(md5, md5_os.finalize());
		char *write = str;
		for(unsigned i = 0; i < 16; ++i){
			*(write++) = HEX_TABLE[md5[i] / 16];
			*(write++) = HEX_TABLE[md5[i] % 16];
		}
		*write = 0;
	}
}

// Digest
std::pair<AuthenticationResult, const char *> check_authentication_digest(
	const boost::shared_ptr<const AuthenticationContext> &context, const IpPort &remote_info, Verb verb, const std::string &header_value)
{
	PROFILE_ME;

	if(!context){
		LOG_POSEIDON_INFO("HTTP authentication succeeded (assuming anonymous).");
		return std::make_pair(AUTH_SUCCEEDED, NULLPTR);
	}
	if(header_value.empty()){
		return std::make_pair(AUTH_HEADER_NOT_SET, NULLPTR);
	}
	if(::strncasecmp(header_value.c_str(), "Digest ", 7) != 0){
		LOG_POSEIDON_WARNING("HTTP authentication scheme not supported: ", header_value);
		return std::make_pair(AUTH_SCHEME_NOT_SUPPORTED, NULLPTR);
	}

	Poseidon::OptionalMap params;
	Buffer_istream bis;
	bis.set_buffer(StreamBuffer(header_value.c_str() + 7));
	std::string seg;
	for(;;){
		seg.clear();
		bool quoted = false;
		bool escaped = false;
		char ch;
		while(bis.get(ch)){
			if(quoted){
				if(escaped){
					seg += ch;
					escaped = false;
					continue;
				}
				if(ch == '\\'){
					escaped = true;
					continue;
				}
				if(ch == '\"'){
					quoted = false;
					continue;
				}
				seg += ch;
				continue;
			}
			if(ch == '\"'){
				quoted = true;
				continue;
			}
			if(ch == ','){
				break;
			}
			seg += ch;
		}
		if(seg.empty()){
			break;
		}
		LOG_POSEIDON_TRACE("> Parsing: ", seg);

		std::size_t key_begin = seg.find_first_not_of(" \t");
		if(key_begin == std::string::npos){
			continue;
		}
		std::size_t equ = seg.find('=', key_begin);
		if(equ == std::string::npos){
			LOG_POSEIDON_WARNING("Invalid HTTP Digest authentication header, equans sign not found: ", seg);
			return std::make_pair(AUTH_HEADER_FORMAT_ERROR, NULLPTR);
		}
		std::size_t key_end;
		if(equ == std::string::npos){
			key_end = seg.find_last_not_of(" \t");
		} else {
			key_end = seg.find_last_not_of(" \t", equ - 1);
		}
		if((key_end == std::string::npos) || (key_begin > key_end)){
			continue;
		}
		++key_end;
		SharedNts key(seg.data() + key_begin, static_cast<std::size_t>(key_end - key_begin));
		if(equ == std::string::npos){
			seg.clear();
		} else {
			seg.erase(0, equ + 1);
		}
		std::string value(trim(STD_MOVE(seg)));
		LOG_POSEIDON_DEBUG("> Digest parameter: ", key, " = ", value);
		params.append(STD_MOVE(key), STD_MOVE(value));
	}

	const AUTO_REF(response_str, params.get("response"));
	if(response_str.empty()){
		LOG_POSEIDON_DEBUG("No digest response set?");
		return std::make_pair(AUTH_HEADER_FORMAT_ERROR, NULLPTR);
	}
	const AUTO_REF(algorithm_str, params.get("algorithm"));
	if(!algorithm_str.empty() && (::strcasecmp(algorithm_str.c_str(), "md5") != 0)){
		LOG_POSEIDON_DEBUG("Unsupported algorithm: ", algorithm_str);
		return std::make_pair(AUTH_ALGORITHM_NOT_SUPPORTED, NULLPTR);
	}
	const AUTO_REF(qop_str, params.get("qop"));
	if(!qop_str.empty() && (::strcasecmp(qop_str.c_str(), "auth") != 0)){
		LOG_POSEIDON_DEBUG("Unsupported QoP: ", qop_str);
		return std::make_pair(AUTH_QOP_NOT_SUPPORTED, NULLPTR);
	}

	Nonce nonce[1];
	const AUTO_REF(nonce_str, params.get("nonce"));
	if(!decrypt_nonce(nonce, nonce_str.c_str(), remote_info.ip())){
		LOG_POSEIDON_DEBUG("Failed to decrypt nonce: ", nonce_str);
		return std::make_pair(AUTH_HEADER_FORMAT_ERROR, NULLPTR);
	}
	if(nonce->server_id != g_server_id){
		LOG_POSEIDON_DEBUG("Server ID mismatch: ", std::hex, std::setfill('0'), std::setw(8), nonce->server_id);
		return std::make_pair(AUTH_PASSWORD_INCORRECT, NULLPTR);
	}
	const AUTO(nonce_expiry_time, MainConfig::get<boost::uint64_t>("http_digest_nonce_expiry_time", 60000));
	if(nonce->timestamp < saturated_sub(get_utc_time(), nonce_expiry_time)){
		LOG_POSEIDON_DEBUG("Nonce expired: ", nonce->timestamp);
		return std::make_pair(AUTH_REQUEST_EXPIRED, NULLPTR);
	}
	const AUTO_REF(username_str, params.get("username"));
	const AUTO(pair, context->get_password(username_str.c_str()));
	if(!pair.second){
		LOG_POSEIDON_DEBUG("User not found: ", username_str);
		return std::make_pair(AUTH_PASSWORD_INCORRECT, NULLPTR);
	}
	const AUTO_REF(realm_str, params.get("realm"));
	const AUTO_REF(cnonce_str, params.get("cnonce"));
	const AUTO_REF(uri_str, params.get("uri"));
	const AUTO_REF(nc_str, params.get("nc"));

	Md5_ostream ha_md5_os, resp_md5_os;
	char str[33];
	// HA1 = MD5(username:realm:password)
	ha_md5_os <<username_str <<':' <<realm_str <<':' <<pair.second;
	finalize_as_hex(str, ha_md5_os);
	// Put the first and the second segments: response = MD5(HA1:nonce: ...
	resp_md5_os <<str <<':' <<nonce_str <<':';
	// HA2 = MD5(verb:uri)
	ha_md5_os <<get_string_from_verb(verb) <<':' <<uri_str;
	finalize_as_hex(str, ha_md5_os);
	// Put segments in the middle if QoP is requested: response = ... nc:cnonce:qop: ...
	if(!qop_str.empty()){
		resp_md5_os <<nc_str <<':' <<cnonce_str <<':' <<qop_str <<':';
	}
	// Put the final segment: response = ... HA2)
	resp_md5_os <<str;
	finalize_as_hex(str, resp_md5_os);
	if(::strcasecmp(response_str.c_str(), str) != 0){
		LOG_POSEIDON_DEBUG("Digest incorrect: ", response_str, ", expecting ", str);
		return std::make_pair(AUTH_PASSWORD_INCORRECT, NULLPTR);
	}
	LOG_POSEIDON_INFO("HTTP authentication succeeded (using password via the Digest scheme): ", pair.first);
	return std::make_pair(AUTH_SUCCEEDED, pair.first);
}
__attribute__((__noreturn__)) void throw_authentication_failure_digest(
	const boost::shared_ptr<const AuthenticationContext> &context, bool is_proxy, const IpPort &remote_info, AuthenticationResult result)
{
	PROFILE_ME;
	DEBUG_THROW_ASSERT(result != AUTH_SUCCEEDED);

	Buffer_ostream bos;
	bos <<"Digest realm=" <<StringQuoter(context->get_realm().c_str());

	Nonce nonce[1];
	create_nonce(nonce);
	char nonce_str[33];
	encrypt_nonce(nonce_str, nonce, remote_info.ip());
	LOG_POSEIDON_DEBUG("New nonce: ", nonce_str);
	bos <<", nonce=" <<StringQuoter(nonce_str);

	if(result == AUTH_REQUEST_EXPIRED){
		bos <<", stale=true";
	}
	bos <<", qop=" <<StringQuoter("auth");

	do_throw_authentication_failure(is_proxy, bos.get_buffer().dump_string());
}

}
}
