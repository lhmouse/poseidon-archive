// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "authorization.hpp"
#include "exception.hpp"
#include "utilities.hpp"
#include "../singletons/main_config.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../job_base.hpp"
#include "../uuid.hpp"
#include "../time.hpp"
#include "../random.hpp"
#include "../hash.hpp"

namespace Poseidon {

namespace Http {
	namespace {
		struct RawNonce {
			boost::uint64_t timestamp;
			boost::uint64_t random;
			Uuid identifier;
		};

		const AUTO(g_identifier, Uuid::random());

		void xor_nonce(RawNonce &raw_nonce, const char *remote_ip){
			boost::uint32_t temp[2];
			temp[0] = static_cast<boost::uint32_t>(::getpid());
			temp[1] = crc32_hash(remote_ip);
			const AUTO(md5, md5_hash(temp, sizeof(temp)));
			for(std::size_t i = 0; i < sizeof(raw_nonce); ++i){
				reinterpret_cast<unsigned char (&)[sizeof(raw_nonce)]>(raw_nonce)[i] ^= md5.at(i % 16);
			}
		}
	}

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
		AUTO(str, auth_header.substr(0, pos));
		if(::strcasecmp(str.c_str(), "Basic") == 0){
			str = base64_decode(auth_header.substr(pos + 1));

			const AUTO(auth_it, std::lower_bound(auth_info->basic_user_pass.begin(), auth_info->basic_user_pass.end(), str));
			if((auth_it == auth_info->basic_user_pass.end()) || (*auth_it != str)){
				LOG_POSEIDON_INFO("> Failed");
				return std::make_pair(AUTH_INVALID_USER_PASS, NULLPTR);
			}
			LOG_POSEIDON_INFO("> Succeeded");
			return std::make_pair(AUTH_SUCCEEDED, &*auth_it);
		} else if(::strcasecmp(str.c_str(), "Digest") == 0){
			str = auth_header.substr(pos + 1);

			std::string username, realm, nonce, uri, qop, cnonce, nc, response, algorithm;
			RawNonce raw_nonce = { };

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
				AUTO(nonce_bytes, base64_decode(nonce));	\
				if(nonce_bytes.size() != sizeof(raw_nonce)){	\
					LOG_POSEIDON_WARNING("> Inacceptable nonce.");	\
					return std::make_pair(AUTH_INACCEPTABLE_NONCE, NULLPTR);	\
				}	\
				std::memcpy(&raw_nonce, nonce_bytes.data(), sizeof(raw_nonce));	\
				xor_nonce(raw_nonce, remote_addr.ip.get());	\
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
			if(!(algorithm.empty() || (::strcasecmp(algorithm.c_str(), "MD5") == 0))){
				LOG_POSEIDON_WARNING("> Inacceptable algorithm: ", algorithm);
				return std::make_pair(AUTH_INACCEPTABLE_ALGORITHM, NULLPTR);
			}

			if(raw_nonce.identifier != g_identifier){
				LOG_POSEIDON_WARNING("> Unexpected identifier: ", raw_nonce.identifier, ", expecting ", g_identifier);
				return std::make_pair(AUTH_INACCEPTABLE_NONCE, NULLPTR);
			}
			const AUTO(local_now, get_local_time());
			if(local_now < raw_nonce.timestamp){
				LOG_POSEIDON_WARNING("> Nonce timestamp is in the future.");
				return std::make_pair(AUTH_EXPIRED, NULLPTR);
			}
			const AUTO(nonce_expiry_time, MainConfig::get<boost::uint64_t>("http_digest_nonce_expiry_time", 60000));
			if(local_now - raw_nonce.timestamp > nonce_expiry_time){
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

			std::string a1, a2;

			a1.reserve(255);
			a1 += username;
			a1 += ':';
			a1 += realm;
			a1 += ':';
			a1.append(*auth_it, username.size() + 1, std::string::npos);

			a2.reserve(255);
			a2 += get_string_from_verb(verb);
			a2 += ':';
			a2 += uri;

			std::string str_to_hash;
			AUTO(md5, md5_hash(a1));
			str_to_hash += hex_encode(md5.data(), md5.size(), false);
			str_to_hash += ':';
			str_to_hash += nonce;
			str_to_hash += ':';
			if(::strcasecmp(qop.c_str(), "auth") == 0){
				str_to_hash += nc;
				str_to_hash += ':';
				str_to_hash += cnonce;
				str_to_hash += ':';
				str_to_hash += qop;
				str_to_hash += ':';
			} else if(!qop.empty()){
				LOG_POSEIDON_WARNING("> Inacceptable qop: ", qop);
				return std::make_pair(AUTH_INACCEPTABLE_QOP, NULLPTR);
			}
			md5 = md5_hash(a2);
			str_to_hash += hex_encode(md5.data(), md5.size(), false);
			md5 = md5_hash(str_to_hash);
			const AUTO(response_expecting, hex_encode(md5.data(), md5.size()));
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

		std::string auth;
		auth.reserve(255);
		auth += "Digest realm=\"";
		switch(auth_result){
		case AUTH_REQUIRED:
			auth += "Authorization required";
			break;

		case AUTH_INVALID_HEADER:
			auth += "Invalid HTTP authorization header";
			break;

		case AUTH_UNKNOWN_SCHEME:
			auth += "Unknown HTTP authorization scheme";
			break;

		case AUTH_INVALID_USER_PASS:
			auth += "Invalid username or password";
			break;

		case AUTH_INACCEPTABLE_NONCE:
			auth += "Nonce is not acceptable";
			break;

		case AUTH_EXPIRED:
			auth += "Nonce has expired";
			break;

		case AUTH_INACCEPTABLE_ALGORITHM:
			auth += "Algorithm is not acceptable";
			break;

		case AUTH_INACCEPTABLE_QOP:
			auth += "QoP is not acceptable";
			break;

		default:
			LOG_POSEIDON_ERROR("HTTP authorization error: auth_result = ", auth_result);
			auth += "Internal server error";
			break;
		}
		auth += "\",nonce=\"";
		RawNonce raw_nonce;
		raw_nonce.timestamp = get_local_time();
		raw_nonce.random = random_uint64();
		raw_nonce.identifier = g_identifier;
		xor_nonce(raw_nonce, remote_addr.ip.get());
		auth += base64_encode(&raw_nonce, sizeof(raw_nonce));
		auth += "\",qop-value=\"auth\",algorithm=\"MD5\"";

		headers.set(SharedNts(auth_name), STD_MOVE(auth));
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
		if(result.first == AUTH_SUCCEEDED){
			return result.second;
		}
		throw_unauthorized(result.first, remote_addr, is_proxy, STD_MOVE(headers));
	}
}

}
