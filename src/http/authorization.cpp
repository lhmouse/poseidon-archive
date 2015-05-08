// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

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

		void xorNonce(RawNonce &rawNonce, const char *remoteIp){
			boost::uint32_t temp[2];
			temp[0] = static_cast<boost::uint32_t>(::getpid());
			temp[1] = crc32Sum(remoteIp);
			unsigned char hash[16];
			md5Sum(hash, temp, 8);
			for(std::size_t i = 0; i < sizeof(rawNonce); ++i){
				reinterpret_cast<unsigned char (&)[sizeof(rawNonce)]>(rawNonce)[i] ^= hash[i % 16];
			}
		}
	}

	struct AuthInfo {
		std::vector<std::string> basicUserPass;

		explicit AuthInfo(std::vector<std::string> basicUserPass_)
			: basicUserPass(STD_MOVE(basicUserPass_))
		{
			std::sort(basicUserPass.begin(), basicUserPass.end());
		}
	};

	boost::shared_ptr<const AuthInfo> createAuthInfo(std::vector<std::string> basicUserPass){
		if(basicUserPass.empty()){
			DEBUG_THROW(BasicException, SSLIT("No username:password provided"));
		}
		return boost::make_shared<AuthInfo>(STD_MOVE(basicUserPass));
	}

	std::pair<AuthResult, const std::string *> checkAuthorizationHeader(
		const boost::shared_ptr<const AuthInfo> &authInfo, const IpPort &remoteAddr, Verb verb, const std::string &authHeader)
	{
		PROFILE_ME;
		LOG_POSEIDON_INFO("Checking HTTP authorization header: ", authHeader);

		if(authHeader.empty()){
			return std::make_pair(AUTH_REQUIRED, NULLPTR);
		}

		const std::size_t pos = authHeader.find(' ');
		if(pos == std::string::npos){
			return std::make_pair(AUTH_INVALID_HEADER, NULLPTR);
		}
		AUTO(str, authHeader.substr(0, pos));
		if(::strcasecmp(str.c_str(), "Basic") == 0){
			str = base64Decode(authHeader.substr(pos + 1));

			const AUTO(authIt, std::lower_bound(authInfo->basicUserPass.begin(), authInfo->basicUserPass.end(), str));
			if((authIt == authInfo->basicUserPass.end()) || (*authIt != str)){
				LOG_POSEIDON_INFO("> Failed");
				return std::make_pair(AUTH_INVALID_USER_PASS, NULLPTR);
			}
			LOG_POSEIDON_INFO("> Succeeded");
			return std::make_pair(AUTH_SUCCEEDED, &*authIt);
		} else if(::strcasecmp(str.c_str(), "Digest") == 0){
			str = authHeader.substr(pos + 1);

			std::string username, realm, nonce, uri, qop, cnonce, nc, response, algorithm;
			RawNonce rawNonce = { };

			enum ParserState {
				PS_KEY_INDENT		= 0,
				PS_KEY				= 1,
				PS_VALUE_INDENT		= 2,
				PS_QUOTED_VALUE		= 3,
				PS_VALUE			= 4,
			} ps = PS_KEY_INDENT;

			std::string key, value;

#define COMMIT_KEY_VALUE	\
			if(::strcasecmp(key.c_str(), "username") == 0){	\
				username = STD_MOVE(value);	\
			} else if(::strcasecmp(key.c_str(), "realm") == 0){	\
				realm = STD_MOVE(value);	\
			} else if(::strcasecmp(key.c_str(), "nonce") == 0){	\
				nonce = STD_MOVE(value);	\
				AUTO(nonceBytes, base64Decode(nonce));	\
				if(nonceBytes.size() != sizeof(rawNonce)){	\
					LOG_POSEIDON_WARNING("> Inacceptable nonce.");	\
					return std::make_pair(AUTH_INACCEPTABLE_NONCE, NULLPTR);	\
				}	\
				std::memcpy(&rawNonce, nonceBytes.data(), sizeof(rawNonce));	\
				xorNonce(rawNonce, remoteAddr.ip.get());	\
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
				LOG_POSEIDON_WARNING("> Error parsing HTTP authorizaiton header: ", authHeader, ", ps = ", ps);
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

			if(rawNonce.identifier != g_identifier){
				LOG_POSEIDON_WARNING("> Unexpected identifier: ", rawNonce.identifier, ", expecting ", g_identifier);
				return std::make_pair(AUTH_INACCEPTABLE_NONCE, NULLPTR);
			}
			const AUTO(localNow, getLocalTime());
			if(localNow < rawNonce.timestamp){
				LOG_POSEIDON_WARNING("> Nonce timestamp is in the future.");
				return std::make_pair(AUTH_EXPIRED, NULLPTR);
			}
			const AUTO(nonceExpiryTime, MainConfig::getConfigFile().get<boost::uint64_t>("http_digest_nonce_expiry_time", 60000));
			if(localNow - rawNonce.timestamp > nonceExpiryTime){
				LOG_POSEIDON_WARNING("> Nonce has expired.");
				return std::make_pair(AUTH_EXPIRED, NULLPTR);
			}

			const AUTO(authIt, std::lower_bound(authInfo->basicUserPass.begin(), authInfo->basicUserPass.end(), username));
			if((authIt == authInfo->basicUserPass.end()) || (authIt->size() < username.size()) ||
				(authIt->compare(0, username.size(), username) != 0) || ((*authIt)[username.size()] != ':'))
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
			a1.append(*authIt, username.size() + 1, std::string::npos);

			a2.reserve(255);
			a2 += getStringFromVerb(verb);
			a2 += ':';
			a2 += uri;

			unsigned char digest[16];
			std::string strToHash;
			md5Sum(digest, a1);
			strToHash += hexEncode(digest, sizeof(digest), false);
			strToHash += ':';
			strToHash += nonce;
			strToHash += ':';
			if(::strcasecmp(qop.c_str(), "auth") == 0){
				strToHash += nc;
				strToHash += ':';
				strToHash += cnonce;
				strToHash += ':';
				strToHash += qop;
				strToHash += ':';
			} else if(!qop.empty()){
				LOG_POSEIDON_WARNING("> Inacceptable qop: ", qop);
				return std::make_pair(AUTH_INACCEPTABLE_QOP, NULLPTR);
			}
			md5Sum(digest, a2);
			strToHash += hexEncode(digest, sizeof(digest), false);
			md5Sum(digest, strToHash);
			const AUTO(responseExpecting, hexEncode(digest, sizeof(digest)));
			LOG_POSEIDON_DEBUG("> Response expecting: ", responseExpecting);
			if(::strcasecmp(response.c_str(), responseExpecting.c_str()) != 0){
				LOG_POSEIDON_WARNING("> Digest mismatch.");
				return std::make_pair(AUTH_INVALID_USER_PASS, NULLPTR);
			}
			LOG_POSEIDON_INFO("> Succeeded");
			return std::make_pair(AUTH_SUCCEEDED, &*authIt);
		}
		LOG_POSEIDON_WARNING("> Unknown HTTP authorization scheme: ", str);
		return std::make_pair(AUTH_UNKNOWN_SCHEME, NULLPTR);
	}
	void throwUnauthorized(AuthResult authResult, const IpPort &remoteAddr, bool isProxy, OptionalMap headers){
		PROFILE_ME;

		const StatusCode statusCode = isProxy ? ST_PROXY_AUTH_REQUIRED : ST_UNAUTHORIZED;
		const char *const authName = isProxy ? "Proxy-Authenticate" : "WWW-Authenticate";

		std::string auth;
		auth.reserve(255);
		auth += "Digest realm=\"";
		switch(authResult){
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
			LOG_POSEIDON_ERROR("HTTP authorization error: authResult = ", authResult);
			auth += "Internal server error";
			break;
		}
		auth += "\",nonce=\"";
		RawNonce rawNonce;
		rawNonce.timestamp = getLocalTime();
		rawNonce.random = rand64();
		rawNonce.identifier = g_identifier;
		xorNonce(rawNonce, remoteAddr.ip.get());
		auth += base64Encode(&rawNonce, sizeof(rawNonce));
		auth += "\",qop-value=\"auth\",algorithm=\"MD5\"";

		headers.set(authName, STD_MOVE(auth));
		DEBUG_THROW(Exception, statusCode, STD_MOVE(headers));
	}

	const std::string *checkAndThrowIfUnauthorized(
		const boost::shared_ptr<const AuthInfo> &authInfo, const IpPort &remoteAddr, const RequestHeaders &requestHeaders,
		bool isProxy, OptionalMap headers)
	{
		PROFILE_ME;

		if(!authInfo){
			return NULLPTR;
		}

		const AUTO_REF(authHeader, requestHeaders.headers.get(isProxy ? "Proxy-Authorization" : "Authorization"));
		const AUTO(result, checkAuthorizationHeader(authInfo, remoteAddr, requestHeaders.verb, authHeader));
		if(result.first == AUTH_SUCCEEDED){
			return result.second;
		}
		throwUnauthorized(result.first, remoteAddr, isProxy, STD_MOVE(headers));
	}
}

}
