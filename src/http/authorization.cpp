// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "authorization.hpp"
#include "exception.hpp"
#include "../hex.hpp"
#include "../base64.hpp"
#include "../singletons/main_config.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../job_base.hpp"
#include "../uuid.hpp"
#include "../time.hpp"
#include "../random.hpp"
#include "../md5.hpp"
#include "../stream_buffer.hpp"
#include <openssl/aes.h>

namespace Poseidon {

namespace {
	struct PlainNonce {
		boost::uint64_t random;
		boost::uint64_t timestamp;
		Uuid identifier;
	};
	typedef boost::array<unsigned char, sizeof(PlainNonce)> CipherNonce;

	BOOST_STATIC_ASSERT(sizeof(PlainNonce) % 16 == 0);

	const Uuid g_identifier = Uuid::random();

	CipherNonce encrypt_nonce(const PlainNonce &plain_nonce, const char *remote_ip){
		PROFILE_ME;

		Md5_ostream md5_os;
		md5_os <<g_identifier <<':' <<remote_ip;
		AUTO(md5, md5_os.finalize());
		::AES_KEY aes_key[1];
		if(::AES_set_encrypt_key(md5.data(), 128, aes_key) != 0){
			LOG_POSEIDON_FATAL("::AES_set_encrypt_key() failed!");
			std::abort();
		}
		CipherNonce cipler_nonce;
		for(std::size_t i = 0; i < sizeof(PlainNonce); i += 16){
			::AES_ecb_encrypt(reinterpret_cast<const unsigned char *>(&plain_nonce) + i, cipler_nonce.data() + i, aes_key, AES_ENCRYPT);
		}
		return cipler_nonce;
	}
	PlainNonce decrypt_nonce(const CipherNonce &cipler_nonce, const char *remote_ip){
		PROFILE_ME;

		Md5_ostream md5_os;
		md5_os <<g_identifier <<':' <<remote_ip;
		AUTO(md5, md5_os.finalize());
		::AES_KEY aes_key[1];
		if(::AES_set_decrypt_key(md5.data(), 128, aes_key) != 0){
			LOG_POSEIDON_FATAL("::AES_set_decrypt_key() failed!");
			std::abort();
		}
		PlainNonce plain_nonce;
		for(std::size_t i = 0; i < sizeof(PlainNonce); i += 16){
			::AES_ecb_encrypt(cipler_nonce.data() + i, reinterpret_cast<unsigned char *>(&plain_nonce) + i, aes_key, AES_DECRYPT);
		}
		return plain_nonce;
	}
}

namespace Http {
	struct AuthInfo {
		std::vector<std::string> basic_user_pass;

		explicit AuthInfo(std::vector<std::string> basic_user_pass_)
			: basic_user_pass(STD_MOVE(basic_user_pass_))
		{
			std::sort(basic_user_pass.begin(), basic_user_pass.end());
		}
	};

	boost::shared_ptr<const AuthInfo> create_auth_info(std::vector<std::string> basic_user_pass){
		if(basic_user_pass.empty()){
			DEBUG_THROW(BasicException, sslit("No username:password provided"));
		}
		return boost::make_shared<AuthInfo>(STD_MOVE(basic_user_pass));
	}

	std::pair<AuthResult, const std::string *> check_authorization_header(
		const boost::shared_ptr<const AuthInfo> &auth_info, const IpPort &remote_addr, Verb verb, const std::string &auth_header)
	{
		PROFILE_ME;
		LOG_POSEIDON_INFO("Checking HTTP authorization header: ", auth_header);

		if(auth_header.empty()){
			return std::make_pair(AUTH_REQUIRED, NULLPTR);
		}

		const AUTO(pos, auth_header.find(' '));
		if(pos == std::string::npos){
			return std::make_pair(AUTH_INVALID_HEADER, NULLPTR);
		}
		std::string str;
		str.reserve(auth_header.size());
		str.assign(auth_header, 0, pos);
		if(::strcasecmp(str.c_str(), "Basic") == 0){
			str.assign(auth_header, pos + 1, std::string::npos);

			Base64Decoder dec;
			dec.put(str);
			str = dec.finalize().dump_string();

			const AUTO(auth_it, std::lower_bound(auth_info->basic_user_pass.begin(), auth_info->basic_user_pass.end(), str));
			if((auth_it == auth_info->basic_user_pass.end()) || (*auth_it != str)){
				LOG_POSEIDON_INFO("> Failed");
				return std::make_pair(AUTH_INVALID_USER_PASS, NULLPTR);
			}
			LOG_POSEIDON_INFO("> Succeeded");
			return std::make_pair(AUTH_SUCCEEDED, &*auth_it);
		} else if(::strcasecmp(str.c_str(), "Digest") == 0){
			str.assign(auth_header, pos + 1, std::string::npos);

			std::string username, realm, nonce, uri, qop, cnonce, nc, response, algorithm;
			PlainNonce plain_nonce = { };

			enum ParserState {
				PS_KEY_INDENT       = 0,
				PS_KEY              = 1,
				PS_VALUE_INDENT     = 2,
				PS_QUOTED_VALUE     = 3,
				PS_VALUE            = 4,
			} ps = PS_KEY_INDENT;

			std::string key, value;

#define COMMIT_KEY_VALUE	\
			if(::strcasecmp(key.c_str(), "username") == 0){	\
				username = STD_MOVE(value);	\
			} else if(::strcasecmp(key.c_str(), "realm") == 0){	\
				realm = STD_MOVE(value);	\
			} else if(::strcasecmp(key.c_str(), "nonce") == 0){	\
				nonce = STD_MOVE(value);	\
				Base64Decoder dec;	\
				dec.put(nonce.data(), nonce.size());	\
				AUTO(buffer, dec.finalize());	\
				CipherNonce cipler_nonce;	\
				if(buffer.get(cipler_nonce.data(), cipler_nonce.size()) < cipler_nonce.size()){	\
					LOG_POSEIDON_WARNING("> Inacceptable nonce.");	\
					return std::make_pair(AUTH_INACCEPTABLE_NONCE, NULLPTR);	\
				}	\
				plain_nonce = decrypt_nonce(cipler_nonce, remote_addr.ip());	\
			} else if(::strcasecmp(key.c_str(), "uri") == 0){	\
				uri = STD_MOVE(value);	\
			} else if(::strcasecmp(key.c_str(), "qop") == 0){	\
				qop = STD_MOVE(value);	\
			} else if(::strcasecmp(key.c_str(), "cnonce") == 0){	\
				cnonce = STD_MOVE(value);	\
			} else if(::strcasecmp(key.c_str(), "nc") == 0){	\
				nc = STD_MOVE(value);	\
			} else if(::strcasecmp(key.c_str(), "response") == 0){	\
				response = STD_MOVE(value);	\
			} else if(::strcasecmp(key.c_str(), "algorithm") == 0){	\
				algorithm = STD_MOVE(value);	\
			}

			for(AUTO(it, str.begin()); it != str.end(); ++it){
				switch(ps){
				case PS_KEY_INDENT:
					if(*it == ' '){
						// ps = PS_KEY_INDENT;
					} else {
						key += *it;
						ps = PS_KEY;
					}
					break;
				case PS_KEY:
					if(*it == '='){
						ps = PS_VALUE_INDENT;
					} else {
						key += *it;
						// ps = PS_KEY;
					}
					break;
				case PS_VALUE_INDENT:
					if(*it == ' '){
						// ps = PS_VALUE_INDENT;
					} else if(*it == '\"'){
						ps = PS_QUOTED_VALUE;
					} else {
						value += *it;
						ps = PS_VALUE;
					}
					break;
				case PS_VALUE:
					if(*it == ','){
						COMMIT_KEY_VALUE;

						key.clear();
						value.clear();
						ps = PS_KEY_INDENT;
					} else {
						value += *it;
						// ps = PS_VALUE;
					}
					break;
				case PS_QUOTED_VALUE:
					if(*it == '\"'){
						ps = PS_VALUE;
					} else {
						value += *it;
						// ps = PS_QUOTED_VALUE;
					}
					break;
				}
			}
			if(ps == PS_VALUE){
				COMMIT_KEY_VALUE;
			} else if(ps != PS_KEY_INDENT){
				LOG_POSEIDON_WARNING("> Error parsing HTTP authorizaiton header: ", auth_header, ", ps = ", ps);
				return std::make_pair(AUTH_INVALID_HEADER, NULLPTR);
			}

			if(username.empty()){
				LOG_POSEIDON_WARNING("> No username specified.");
				return std::make_pair(AUTH_INVALID_USER_PASS, NULLPTR);
			}
			if(nonce.empty()){
				LOG_POSEIDON_WARNING("> No nonce specified.");
				return std::make_pair(AUTH_INACCEPTABLE_NONCE, NULLPTR);
			}
			if(!(qop.empty() || (::strcasecmp(qop.c_str(), "auth") == 0))){
				LOG_POSEIDON_WARNING("> Inacceptable qop: ", qop);
				return std::make_pair(AUTH_INACCEPTABLE_QOP, NULLPTR);
			}
			if(!(algorithm.empty() || (::strcasecmp(algorithm.c_str(), "MD5") == 0))){
				LOG_POSEIDON_WARNING("> Inacceptable algorithm: ", algorithm);
				return std::make_pair(AUTH_INACCEPTABLE_ALGORITHM, NULLPTR);
			}

			if(plain_nonce.identifier != g_identifier){
				LOG_POSEIDON_WARNING("> Unexpected identifier: ", plain_nonce.identifier, ", expecting ", g_identifier);
				return std::make_pair(AUTH_INACCEPTABLE_NONCE, NULLPTR);
			}
			const AUTO(local_now, get_local_time());
			if(local_now < plain_nonce.timestamp){
				LOG_POSEIDON_WARNING("> Nonce timestamp is in the future.");
				return std::make_pair(AUTH_EXPIRED, NULLPTR);
			}
			const AUTO(nonce_expiry_time, MainConfig::get<boost::uint64_t>("http_digest_nonce_expiry_time", 60000));
			if(local_now - plain_nonce.timestamp > nonce_expiry_time){
				LOG_POSEIDON_WARNING("> Nonce has expired.");
				return std::make_pair(AUTH_EXPIRED, NULLPTR);
			}

			const AUTO(auth_it, std::lower_bound(auth_info->basic_user_pass.begin(), auth_info->basic_user_pass.end(), username));
			if((auth_it == auth_info->basic_user_pass.end()) || (auth_it->size() < username.size()) ||
				(auth_it->compare(0, username.size(), username) != 0) || ((*auth_it)[username.size()] != ':'))
			{
				LOG_POSEIDON_WARNING("> Username not found: ", username);
				return std::make_pair(AUTH_INVALID_USER_PASS, NULLPTR);
			}

			Md5_ostream a1_md5s, a2_md5s;
			a1_md5s <<username <<':' <<realm <<':' <<(auth_it->data() + username.size() + 1);
			a2_md5s <<get_string_from_verb(verb) <<':' <<uri;

			Md5_ostream resp_md5s;
			AUTO(md5, a1_md5s.finalize());
			HexEncoder enc;
			enc.put(md5.data(), md5.size());
			resp_md5s <<enc.finalize() <<':' <<nonce <<':';
			if(!qop.empty()){
				resp_md5s <<nc <<':' <<cnonce <<':' <<qop <<':';
			}
			md5 = a2_md5s.finalize();
			enc.put(md5.data(), md5.size());
			resp_md5s <<enc.finalize();
			md5 = resp_md5s.finalize();
			enc.put(md5.data(), md5.size());
			const AUTO(response_expecting, enc.finalize().dump_string());
			LOG_POSEIDON_DEBUG("> Response expecting: ", response_expecting);
			if(::strcasecmp(response.c_str(), response_expecting.c_str()) != 0){
				LOG_POSEIDON_WARNING("> Digest mismatch.");
				return std::make_pair(AUTH_INVALID_USER_PASS, NULLPTR);
			}
			LOG_POSEIDON_INFO("> Succeeded");
			return std::make_pair(AUTH_SUCCEEDED, &*auth_it);
		}
		LOG_POSEIDON_WARNING("> Unknown HTTP authorization scheme: ", str);
		return std::make_pair(AUTH_UNKNOWN_SCHEME, NULLPTR);
	}
	void throw_unauthorized(AuthResult auth_result, const IpPort &remote_addr, bool is_proxy, OptionalMap headers){
		PROFILE_ME;

		const StatusCode status_code = is_proxy ? ST_PROXY_AUTH_REQUIRED : ST_UNAUTHORIZED;
		const char *const auth_name = is_proxy ? "Proxy-Authenticate" : "WWW-Authenticate";

		const char *realm;
		switch(auth_result){
		case AUTH_REQUIRED:
			realm = "Authorization required";
			break;
		case AUTH_INVALID_HEADER:
			realm = "Invalid HTTP authorization header";
			break;
		case AUTH_UNKNOWN_SCHEME:
			realm = "Unknown HTTP authorization scheme";
			break;
		case AUTH_INVALID_USER_PASS:
			realm = "Invalid username or password";
			break;
		case AUTH_INACCEPTABLE_NONCE:
			realm = "Nonce is not acceptable";
			break;
		case AUTH_EXPIRED:
			realm = "Nonce has expired";
			break;
		case AUTH_INACCEPTABLE_ALGORITHM:
			realm = "Algorithm is not acceptable";
			break;
		case AUTH_INACCEPTABLE_QOP:
			realm = "QoP is not acceptable";
			break;
		default:
			LOG_POSEIDON_ERROR("HTTP authorization error: auth_result = ", auth_result);
			realm = "Internal server error";
			break;
		}

		PlainNonce plain_nonce = { };
		plain_nonce.timestamp = get_local_time();
		plain_nonce.random = random_uint64();
		plain_nonce.identifier = g_identifier;
		CipherNonce cipler_nonce;
		cipler_nonce = encrypt_nonce(plain_nonce, remote_addr.ip());
		Base64Encoder enc;
		enc.put(cipler_nonce.data(), cipler_nonce.size());
		AUTO(nonce, enc.finalize());

		StreamBuffer auth;
		auth.put("Digest realm=\"");
		auth.put(realm);
		auth.put("\", nonce=\"");
		auth.splice(nonce);
		auth.put("\", qop-value=\"auth\", algorithm=\"MD5\"");
		headers.set(SharedNts(auth_name), auth.dump_string());
		DEBUG_THROW(Exception, status_code, STD_MOVE(headers));
	}

	const std::string *check_and_throw_if_unauthorized(
		const boost::shared_ptr<const AuthInfo> &auth_info, const IpPort &remote_addr, const RequestHeaders &request_headers,
		bool is_proxy,
#ifdef POSEIDON_CXX11
		Move<OptionalMap>
#else
		OptionalMap
#endif
			headers)
	{
		PROFILE_ME;

		if(!auth_info){
			return NULLPTR;
		}

		const AUTO_REF(auth_header, request_headers.headers.get(is_proxy ? "Proxy-Authorization" : "Authorization"));
		const AUTO(result, check_authorization_header(auth_info, remote_addr, request_headers.verb, auth_header));
		if(result.first != AUTH_SUCCEEDED){
			throw_unauthorized(result.first, remote_addr, is_proxy, STD_MOVE(headers));
			std::abort();
		}
		return result.second;
	}
}

}
