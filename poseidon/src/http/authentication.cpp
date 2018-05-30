// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

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

class Authentication_context {
private:
	struct Password_comparator {
		bool operator()(const std::string &lhs, const char *rhs) const {
			return ::strcasecmp(lhs.c_str(), rhs);
		}
		bool operator()(const char *lhs, const std::string &rhs) const {
			return ::strcasecmp(lhs, rhs.c_str());
		}
	};

private:
	const std::string m_realm;

	boost::container::vector<std::string> m_passwords;

public:
	explicit Authentication_context(std::string realm)
		: m_realm(STD_MOVE(realm))
	{
		//
	}

public:
	const std::string & get_realm() const {
		return m_realm;
	}

	std::pair<const char *, const char *> get_password(const char *username) const {
		const AUTO(range, std::equal_range(m_passwords.begin(), m_passwords.end(), username, Password_comparator()));
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
		const AUTO(range, std::equal_range(m_passwords.begin(), m_passwords.end(), username, Password_comparator()));
		if(range.first == range.second){
			m_passwords.insert(range.first, STD_MOVE(str));
		} else {
			range.first->swap(str);
		}
	}
};

boost::shared_ptr<const Authentication_context> create_authentication_context(const std::string &realm, const boost::container::vector<std::string> &basic_user_pass){
	POSEIDON_PROFILE_ME;
	POSEIDON_THROW_UNLESS(!basic_user_pass.empty(), Basic_exception, Rcnts::view("No username:password provided"));

	AUTO(context, boost::make_shared<Authentication_context>(realm));
	std::string str;
	for(AUTO(it, basic_user_pass.begin()); it != basic_user_pass.end(); ++it){
		str = *it;
		AUTO(pos, str.find('\0'));
		POSEIDON_THROW_UNLESS(pos == std::string::npos, Basic_exception, Rcnts::view("Username or password shall not contain null characters"));
		pos = str.find(':');
		POSEIDON_THROW_UNLESS(pos != std::string::npos, Basic_exception, Rcnts::view("Colon delimiter not found"));
		str.at(pos) = 0;
		const AUTO(old_password, context->get_password(str.c_str()).second);
		POSEIDON_THROW_UNLESS(!old_password, Basic_exception, Rcnts::view("Duplicate username"));
		context->set_password(str.c_str(), str.c_str() + pos + 1);
	}
	return STD_MOVE_IDN(context);
}
std::pair<Authentication_result, const char *> check_authentication(const boost::shared_ptr<const Authentication_context> &context, bool is_proxy, const Ip_port &remote_info, const Request_headers &request_headers){
	POSEIDON_PROFILE_ME;

	if(!context){
		POSEIDON_LOG_INFO("HTTP authentication succeeded (assuming anonymous).");
		return std::make_pair(auth_succeeded, NULLPTR);
	}
	const AUTO_REF(header_value, request_headers.headers.get(is_proxy ? "Proxy-Authorization" : "Authorization"));
	if(header_value.empty()){
		return std::make_pair(auth_header_not_set, NULLPTR);
	}
	if(::strncasecmp(header_value.c_str(), "Basic ", 6) == 0){
		return check_authentication_basic(context, header_value);
	}
	if(::strncasecmp(header_value.c_str(), "Digest ", 7) == 0){
		return check_authentication_digest(context, remote_info, request_headers.verb, header_value);
	}
	POSEIDON_LOG_WARNING("HTTP authentication scheme not supported: ", header_value);
	return std::make_pair(auth_scheme_not_supported, NULLPTR);
}
__attribute__((__noreturn__)) void throw_authentication_failure(const boost::shared_ptr<const Authentication_context> &context, bool is_proxy, const Ip_port &remote_info, Authentication_result result){
	POSEIDON_PROFILE_ME;

	throw_authentication_failure_digest(context, is_proxy, remote_info, result);
}
const char * check_authentication_simple(const boost::shared_ptr<const Authentication_context> &context, bool is_proxy, const Ip_port &remote_info, const Request_headers &request_headers){
	POSEIDON_PROFILE_ME;

	const AUTO(pair, check_authentication(context, is_proxy, remote_info, request_headers));
	if(pair.first != auth_succeeded){
		POSEIDON_THROW_ASSERT(context);
		throw_authentication_failure(context, is_proxy, remote_info, pair.first);
	}
	return pair.second;
}

namespace {
	class String_quoter {
	private:
		const char *m_str;

	public:
		String_quoter(const char *str)
			: m_str(str)
		{
			//
		}

	public:
		const char * get() const {
			return m_str;
		}
	};

	std::ostream & operator<<(std::ostream &os, const String_quoter &rhs){
		POSEIDON_PROFILE_ME;

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
		POSEIDON_PROFILE_ME;

		Option_map headers;
		headers.set(Rcnts::view(is_proxy ? "Proxy-Authenticate" : "WWW-Authenticate"), STD_MOVE(authenticate_str));
		POSEIDON_THROW(Exception, is_proxy ? status_proxy_auth_required : status_unauthorized, STD_MOVE(headers));
	}
}

// Basic
std::pair<Authentication_result, const char *> check_authentication_basic(const boost::shared_ptr<const Authentication_context> &context, const std::string &header_value){
	POSEIDON_PROFILE_ME;

	if(!context){
		POSEIDON_LOG_INFO("HTTP authentication succeeded (assuming anonymous).");
		return std::make_pair(auth_succeeded, NULLPTR);
	}
	if(header_value.empty()){
		return std::make_pair(auth_header_not_set, NULLPTR);
	}
	if(::strncasecmp(header_value.c_str(), "Basic ", 6) != 0){
		POSEIDON_LOG_WARNING("HTTP authentication scheme not supported: ", header_value);
		return std::make_pair(auth_scheme_not_supported, NULLPTR);
	}

	AUTO(auth_str, base64_decode(header_value.c_str() + 6));
	const AUTO(colon_pos, auth_str.find(':'));
	if(colon_pos == std::string::npos){
		POSEIDON_LOG_ERROR("Colon delimiter not found: ", auth_str);
		return std::make_pair(auth_header_format_error, NULLPTR);
	}
	auth_str.at(colon_pos) = 0;

	const AUTO(pair, context->get_password(auth_str.c_str()));
	if(!pair.second){
		POSEIDON_LOG_DEBUG("User not found: ", auth_str);
		return std::make_pair(auth_password_incorrect, NULLPTR);
	}
	if(::strcmp(auth_str.c_str() + colon_pos + 1, pair.second) != 0){
		POSEIDON_LOG_DEBUG("Password incorrect: ", auth_str);
		return std::make_pair(auth_password_incorrect, NULLPTR);
	}
	POSEIDON_LOG_INFO("HTTP authentication succeeded (using password via the Basic scheme): ", pair.first);
	return std::make_pair(auth_succeeded, pair.first);
}
__attribute__((__noreturn__)) void throw_authentication_failure_basic(const boost::shared_ptr<const Authentication_context> &context, bool is_proxy, Authentication_result result){
	POSEIDON_PROFILE_ME;
	POSEIDON_THROW_ASSERT(result != auth_succeeded);

	Buffer_ostream bos;
	bos <<"Basic realm=" <<String_quoter(context->get_realm().c_str());

	do_throw_authentication_failure(is_proxy, bos.get_buffer().dump_string());
}

namespace {
	const std::uint32_t g_server_id = random_uint32();

	struct Nonce {
		std::uint32_t server_id; // g_server_id
		std::uint32_t reserved;
		std::uint64_t timestamp; // get_utc_time()
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
		POSEIDON_THROW_ASSERT(err_code == 0);
		std::array<unsigned char, 16> out;
		::AES_encrypt(reinterpret_cast<const unsigned char *>(nonce), out.data(), &aes_key);
		char *write = str;
		for(unsigned i = 0; i < 16; ++i){
			const int hi = out[i] / 16;
			const int lo = out[i] % 16;
			*(write++) = static_cast<char>('a' + hi);
			*(write++) = static_cast<char>('a' + lo);
		}
		*write = 0;
	}
	bool decrypt_nonce(Nonce *nonce, const char *str, const char *key){
		Md5_ostream md5_os;
		md5_os <<key;
		AUTO(md5, md5_os.finalize());
		::AES_KEY aes_key;
		int err_code = ::AES_set_decrypt_key(md5.data(), 128, &aes_key);
		POSEIDON_THROW_ASSERT(err_code == 0);
		std::array<unsigned char, 16> in;
		const char *read = str;
		for(unsigned i = 0; i < 16; ++i){
			const int hi = *(read++) - 'a';
			if((hi < 0) || (hi >= 16)){
				return false;
			}
			const int lo = *(read++) - 'a';
			if((lo < 0) || (lo >= 16)){
				return false;
			}
			in[i] = static_cast<unsigned char>(hi * 16 + lo);
		}
		if(*read != 0){
			return false;
		}
		::AES_decrypt(in.data(), reinterpret_cast<unsigned char *>(nonce), &aes_key);
		return true;
	}

	void finalize_as_hex(char *str, Md5_ostream &md5_os){
		static CONSTEXPR const char s_hex_table[] = "0123456789abcdef";
		const AUTO(md5, md5_os.finalize());
		char *write = str;
		for(unsigned i = 0; i < 16; ++i){
			*(write++) = s_hex_table[md5[i] / 16];
			*(write++) = s_hex_table[md5[i] % 16];
		}
		*write = 0;
	}
}

// Digest
std::pair<Authentication_result, const char *> check_authentication_digest(const boost::shared_ptr<const Authentication_context> &context, const Ip_port &remote_info, Verb verb, const std::string &header_value){
	POSEIDON_PROFILE_ME;

	if(!context){
		POSEIDON_LOG_INFO("HTTP authentication succeeded (assuming anonymous).");
		return std::make_pair(auth_succeeded, NULLPTR);
	}
	if(header_value.empty()){
		return std::make_pair(auth_header_not_set, NULLPTR);
	}
	if(::strncasecmp(header_value.c_str(), "Digest ", 7) != 0){
		POSEIDON_LOG_WARNING("HTTP authentication scheme not supported: ", header_value);
		return std::make_pair(auth_scheme_not_supported, NULLPTR);
	}

	Option_map params;
	Buffer_istream bis;
	bis.set_buffer(Stream_buffer(header_value.c_str() + 7));
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
		POSEIDON_LOG_TRACE("> Parsing: ", seg);

		std::size_t key_begin = seg.find_first_not_of(" \t");
		if(key_begin == std::string::npos){
			continue;
		}
		std::size_t equ = seg.find('=', key_begin);
		if(equ == std::string::npos){
			POSEIDON_LOG_WARNING("Invalid HTTP Digest authentication header, equals sign not found: ", seg);
			return std::make_pair(auth_header_format_error, NULLPTR);
		}
		std::size_t key_end = seg.find_last_not_of(" \t", equ - 1);
		if((key_end == std::string::npos) || (key_begin > key_end)){
			POSEIDON_LOG_WARNING("Invalid HTTP Digest authentication header, no key specified: ", seg);
			return std::make_pair(auth_header_format_error, NULLPTR);
		}
		++key_end;
		Rcnts key(seg.data() + key_begin, static_cast<std::size_t>(key_end - key_begin));
		if(equ == std::string::npos){
			seg.clear();
		} else {
			seg.erase(0, equ + 1);
		}
		std::string value(trim(STD_MOVE(seg)));
		POSEIDON_LOG_DEBUG("> Digest parameter: ", key, " = ", value);
		params.append(STD_MOVE(key), STD_MOVE(value));
	}

	const AUTO_REF(response_str, params.get("response"));
	if(response_str.empty()){
		POSEIDON_LOG_DEBUG("No digest response set?");
		return std::make_pair(auth_header_format_error, NULLPTR);
	}
	const AUTO_REF(algorithm_str, params.get("algorithm"));
	if(!algorithm_str.empty() && (::strcasecmp(algorithm_str.c_str(), "md5") != 0)){
		POSEIDON_LOG_DEBUG("Unsupported algorithm: ", algorithm_str);
		return std::make_pair(auth_algorithm_not_supported, NULLPTR);
	}
	const AUTO_REF(qop_str, params.get("qop"));
	if(!qop_str.empty() && (::strcasecmp(qop_str.c_str(), "auth") != 0)){
		POSEIDON_LOG_DEBUG("Unsupported QoP: ", qop_str);
		return std::make_pair(auth_qop_not_supported, NULLPTR);
	}

	Nonce nonce[1];
	const AUTO_REF(nonce_str, params.get("nonce"));
	if(!decrypt_nonce(nonce, nonce_str.c_str(), remote_info.ip())){
		POSEIDON_LOG_DEBUG("Failed to decrypt nonce: ", nonce_str);
		return std::make_pair(auth_header_format_error, NULLPTR);
	}
	if(nonce->server_id != g_server_id){
		POSEIDON_LOG_DEBUG("Server ID mismatch: ", std::hex, std::setfill('0'), std::setw(8), nonce->server_id);
		return std::make_pair(auth_password_incorrect, NULLPTR);
	}
	const AUTO(nonce_expiry_time, Main_config::get<std::uint64_t>("http_digest_nonce_expiry_time", 60000));
	if(nonce->timestamp < saturated_sub(get_utc_time(), nonce_expiry_time)){
		POSEIDON_LOG_DEBUG("Nonce expired: ", nonce->timestamp);
		return std::make_pair(auth_request_expired, NULLPTR);
	}
	const AUTO_REF(username_str, params.get("username"));
	const AUTO(pair, context->get_password(username_str.c_str()));
	if(!pair.second){
		POSEIDON_LOG_DEBUG("User not found: ", username_str);
		return std::make_pair(auth_password_incorrect, NULLPTR);
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
		POSEIDON_LOG_DEBUG("Digest incorrect: ", response_str, ", expecting ", str);
		return std::make_pair(auth_password_incorrect, NULLPTR);
	}
	POSEIDON_LOG_INFO("HTTP authentication succeeded (using password via the Digest scheme): ", pair.first);
	return std::make_pair(auth_succeeded, pair.first);
}
__attribute__((__noreturn__)) void throw_authentication_failure_digest(const boost::shared_ptr<const Authentication_context> &context, bool is_proxy, const Ip_port &remote_info, Authentication_result result){
	POSEIDON_PROFILE_ME;
	POSEIDON_THROW_ASSERT(result != auth_succeeded);

	Buffer_ostream bos;
	bos <<"Digest realm=" <<String_quoter(context->get_realm().c_str());

	Nonce nonce[1];
	create_nonce(nonce);
	char nonce_str[33];
	encrypt_nonce(nonce_str, nonce, remote_info.ip());
	POSEIDON_LOG_DEBUG("New nonce: ", nonce_str);
	bos <<", nonce=" <<String_quoter(nonce_str);

	if(result == auth_request_expired){
		bos <<", stale=true";
	}
	bos <<", qop=" <<String_quoter("auth");

	do_throw_authentication_failure(is_proxy, bos.get_buffer().dump_string());
}

}
}
