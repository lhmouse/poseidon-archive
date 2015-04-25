// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "exception.hpp"
#include "utilities.hpp"
#include "upgraded_session_base.hpp"
#include "../log.hpp"
#include "../singletons/main_config.hpp"
#include "../singletons/epoll_daemon.hpp"
#include "../stream_buffer.hpp"
#include "../time.hpp"
#include "../random.hpp"
#include "../string.hpp"
#include "../exception.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"
#include "../hash.hpp"

namespace Poseidon {

namespace Http {
	namespace {
		class SessionJobBase : public JobBase {
		private:
			const boost::weak_ptr<Session> m_session;

		protected:
			explicit SessionJobBase(const boost::shared_ptr<Session> &session)
				: m_session(session)
			{
			}

		protected:
			virtual void perform(const boost::shared_ptr<Session> &session) const = 0;

		private:
			boost::weak_ptr<const void> getCategory() const FINAL {
				return m_session;
			}
			void perform() const FINAL {
				PROFILE_ME;

				const AUTO(session, m_session.lock());
				if(!session){
					return;
				}

				try {
					perform(session);
				} catch(TryAgainLater &){
					throw;
				} catch(std::exception &e){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what());
					session->forceShutdown();
					throw;
				} catch(...){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown.");
					session->forceShutdown();
					throw;
				}
			}
		};

		class ContinueJob : public SessionJobBase {
		public:
			explicit ContinueJob(const boost::shared_ptr<Session> &session)
				: SessionJobBase(session)
			{
			}

		protected:
			void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
				PROFILE_ME;

				session->sendDefault(ST_CONTINUE);
			}
		};

		void xorNonce(void *nonce, std::size_t size, const char *remoteIp){
			boost::uint32_t temp[2];
			storeBe(temp[0], static_cast<boost::uint32_t>(::getpid()));
			storeBe(temp[1], crc32Sum(remoteIp, std::strlen(remoteIp)));
			unsigned char hash[16];
			md5Sum(hash, temp, 8);
			for(std::size_t i = 0; i < size; ++i){
				static_cast<unsigned char *>(nonce)[i] ^= hash[i % 16];
			}
		}

		enum AuthResult {
			AUTH_SUCCESSFUL,
			AUTH_REQUIRING,
			AUTH_INVALID_HEADER,
			AUTH_UNKNOWN_SCHEME,
			AUTH_INVALID_USER_PASS,
			AUTH_INACCEPTABLE_NONCE,
			AUTH_EXPIRED,
			AUTH_INACCEPTABLE_ALGORITHM,
			AUTH_INACCEPTABLE_QOP,
		};

		AuthResult checkAuthorization(const boost::shared_ptr<Session::BasicAuthInfo> &authInfo,
			const char *remoteIp, Verb verb, const std::string &authHeader)
		{
			PROFILE_ME;
			LOG_POSEIDON_INFO("Checking HTTP authorization: ", authHeader);

			const std::size_t pos = authHeader.find(' ');
			if(pos == std::string::npos){
				return AUTH_INVALID_HEADER;
			}
			AUTO(str, authHeader.substr(0, pos));
			if(::strcasecmp(str.c_str(), "Basic") == 0){
				str = base64Decode(authHeader.substr(pos + 1));

				if(!std::binary_search(authInfo->begin(), authInfo->end(), str)){
					LOG_POSEIDON_INFO("> Failed");
					return AUTH_INVALID_USER_PASS;
				}
				LOG_POSEIDON_INFO("> Succeeded");
				return AUTH_SUCCESSFUL;
			} else if(::strcasecmp(str.c_str(), "Digest") == 0){
				str = authHeader.substr(pos + 1);

				std::string username, realm, nonce, uri, qop, cnonce, nc, response, algorithm;
				boost::uint64_t rawNonce[2] = { };

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
					for(AUTO(it, username.begin()); it != username.end(); ++it){	\
						if(*it == ':'){	\
							*it = ' ';	\
						}	\
					}	\
				} else if(::strcasecmp(key.c_str(), "realm") == 0){	\
					realm = STD_MOVE(value);	\
				} else if(::strcasecmp(key.c_str(), "nonce") == 0){	\
					nonce = STD_MOVE(value);	\
					AUTO(nonceBytes, base64Decode(nonce));	\
					if(nonceBytes.size() != sizeof(rawNonce)){	\
						LOG_POSEIDON_WARNING("> Inacceptable nonce.");	\
						return AUTH_INACCEPTABLE_NONCE;	\
					}	\
					std::memcpy(rawNonce, nonceBytes.data(), sizeof(rawNonce));	\
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
					return AUTH_INVALID_HEADER;
				}

				if(username.empty()){
					LOG_POSEIDON_WARNING("> No username specified.");
					return AUTH_INVALID_USER_PASS;
				}
				if(nonce.empty()){
					LOG_POSEIDON_WARNING("> No nonce specified.");
					return AUTH_INACCEPTABLE_NONCE;
				}
				if(!(algorithm.empty() || (::strcasecmp(algorithm.c_str(), "MD5") == 0))){
					LOG_POSEIDON_WARNING("> Inacceptable algorithm: ", algorithm);
					return AUTH_INACCEPTABLE_ALGORITHM;
				}

				xorNonce(&rawNonce, sizeof(rawNonce), remoteIp);
				const AUTO(timestamp, loadBe(rawNonce[0]));
				const AUTO(now, getFastMonoClock());
				if(now < timestamp){
					LOG_POSEIDON_WARNING("> Nonce timestamp is in the future.");
					return AUTH_EXPIRED;
				} else if(now - timestamp > MainConfig::getConfigFile().get<boost::uint64_t>("http_digest_auth_timeout", 60000)){
					LOG_POSEIDON_WARNING("> Nonce has expired.");
					return AUTH_EXPIRED;
				}

				const AUTO(authIt, std::lower_bound(authInfo->begin(), authInfo->end(), username));
				if((authIt == authInfo->end()) || (authIt->size() < username.size()) ||
					(authIt->compare(0, username.size(), username) != 0) || ((*authIt)[username.size()] != ':'))
				{
					LOG_POSEIDON_WARNING("> Username not found: ", username);
					return AUTH_INVALID_USER_PASS;
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
				md5Sum(digest, a1.data(), a1.size());
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
					return AUTH_INACCEPTABLE_QOP;
				}
				md5Sum(digest, a2.data(), a2.size());
				strToHash += hexEncode(digest, sizeof(digest), false);
				md5Sum(digest, strToHash.data(), strToHash.size());
				const AUTO(responseExpecting, hexEncode(digest, sizeof(digest)));
				LOG_POSEIDON_DEBUG("> Response expecting: ", responseExpecting);
				if(::strcasecmp(response.c_str(), responseExpecting.c_str()) != 0){
					LOG_POSEIDON_WARNING("> Digest mismatch.");
					return AUTH_INVALID_USER_PASS;
				}
				LOG_POSEIDON_INFO("> Succeeded");
				return AUTH_SUCCESSFUL;
			}
			LOG_POSEIDON_WARNING("> Unknown HTTP authorization scheme: ", str);
			return AUTH_UNKNOWN_SCHEME;
		}

		class UnauthorizedJob : public SessionJobBase {
		private:
			const AuthResult m_authResult;

		public:
			UnauthorizedJob(const boost::shared_ptr<Session> &session, AuthResult authResult)
				: SessionJobBase(session)
				, m_authResult(authResult)
			{
			}

		protected:
			void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
				PROFILE_ME;

				const char *realm;
				switch(m_authResult){
				case AUTH_REQUIRING:
					realm = "Authorization required";
					break;

				case AUTH_INVALID_HEADER:
					realm = "Invalid Authorization header";
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
					LOG_POSEIDON_ERROR("HTTP authorization error: authResult = ", m_authResult);
					realm = "Internal server error";
					break;
				}

				boost::uint64_t rawNonce[2];
				storeBe(rawNonce[0], getFastMonoClock());
				storeBe(rawNonce[1], rand64());
				xorNonce(rawNonce, sizeof(rawNonce), session->getRemoteInfo().ip.get());
				const AUTO(nonce, base64Encode(&rawNonce, sizeof(rawNonce)));

				std::string auth;
				auth.reserve(255);
				auth += "Digest realm=\"";
				auth += realm;
				auth += "\",nonce=\"";
				auth += nonce;
				auth += "\",qop-value=\"auth\",algorithm=\"MD5\"";

				OptionalMap headers;
				headers.set("WWW-Authenticate", STD_MOVE(auth));
				session->sendDefault(ST_UNAUTHORIZED, STD_MOVE(headers));
			}
		};
	}

	class Session::RequestJob : public SessionJobBase {
	private:
		const Header m_header;
		const StreamBuffer m_entity;

	public:
		RequestJob(const boost::shared_ptr<Session> &session, Header header, StreamBuffer entity)
			: SessionJobBase(session)
			, m_header(STD_MOVE(header)), m_entity(STD_MOVE(entity))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			try {
				LOG_POSEIDON_DEBUG("Dispatching request: URI = ", m_header.uri);

				session->onRequest(m_header, m_entity);

				const AUTO_REF(keepAlive, m_header.headers.get("Connection"));
				if((m_header.version < 10001)
					? (::strcasecmp(keepAlive.c_str(), "Keep-Alive") == 0)	// HTTP 1.0
					: (::strcasecmp(keepAlive.c_str(), "Close") != 0))		// HTTP 1.1
				{
					session->setTimeout(MainConfig::getConfigFile().get<boost::uint64_t>("http_keep_alive_timeout", 0));
				} else {
					session->shutdown();
				}
			} catch(TryAgainLater &){
				throw;
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Exception thrown in HTTP servlet: URI = ", m_header.uri,
					", statusCode = ", e.statusCode());
				try {
					session->sendDefault(e.statusCode(), e.headers(), false); // 不关闭连接。
				} catch(...){
					session->forceShutdown();
				}
			}
		}
	};

	class Session::ErrorJob : public SessionJobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;

		const StatusCode m_statusCode;
		const OptionalMap m_headers;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session, StatusCode statusCode, OptionalMap headers)
			: SessionJobBase(session)
			, m_guard(session)
			, m_statusCode(statusCode), m_headers(STD_MOVE(headers))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->sendDefault(m_statusCode, m_headers, true);
		}
	};

	class Session::UpgradeJob : public SessionJobBase {
	private:
		const Header m_header;
		const StreamBuffer m_entity;

	public:
		UpgradeJob(const boost::shared_ptr<Session> &session, Header header, StreamBuffer entity)
			: SessionJobBase(session)
			, m_header(STD_MOVE(header)), m_entity(STD_MOVE(entity))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			try {
				AUTO(upgradedSession, session->onUpgrade(m_header, m_entity));
				if(!upgradedSession){
					LOG_POSEIDON_ERROR("Upgrade failed.");
					DEBUG_THROW(Exception, ST_BAD_REQUEST);
				}
				{
					const boost::mutex::scoped_lock lock(session->m_upgreadedMutex);
					session->m_upgradedSession = STD_MOVE(upgradedSession);
				}
				LOG_POSEIDON_DEBUG("Upgraded succeeded: remote = ", session->getRemoteInfo());
				session->setTimeout(MainConfig::getConfigFile().get<boost::uint64_t>("epoll_tcp_request_timeout", 0));
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Exception thrown in HTTP servlet: URI = ", m_header.uri,
					", statusCode = ", e.statusCode());
				try {
					session->sendDefault(e.statusCode(), e.headers(), false); // 不关闭连接。
				} catch(...){
					session->forceShutdown();
				}
			}
		}
	};

	Session::Session(UniqueFile socket, boost::shared_ptr<Session::BasicAuthInfo> authInfo)
		: TcpSessionBase(STD_MOVE(socket))
		, m_authInfo(STD_MOVE(authInfo))
		, m_sizeTotal(0), m_expectingNewLine(true), m_sizeExpecting(0), m_state(S_FIRST_HEADER)
		, m_header()
	{
		if(m_authInfo){
#ifdef POSEIDON_CXX11
			const bool isSorted = std::is_sorted(m_authInfo->begin(), m_authInfo->end());
#else
			bool isSorted = true;
			if(m_authInfo->size() >= 2){
				for(AUTO(it, m_authInfo->begin() + 1); it != m_authInfo->end(); ++it){
					if(!(it[-1] < it[0])){
						isSorted = false;
						break;
					}
				}
			}
#endif
			if(!isSorted){
				LOG_POSEIDON_ERROR("authInfo is not sorted.");
				DEBUG_THROW(BasicException, SSLIT("authInfo is not sorted"));
			}
		}
	}
	Session::~Session(){
		if((m_state != S_FIRST_HEADER) && (m_state != S_UPGRADED)){
			LOG_POSEIDON_WARNING("Now that this session is to be destroyed, a premature request has to be discarded.");
		}
	}

	void Session::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		if(m_state == S_UPGRADED){
			const AUTO(upgradedSession, getUpgradedSession());
			if(upgradedSession){
				upgradedSession->onReadAvail(data, size);
				return;
			}
			LOG_POSEIDON_WARNING("Session has not been fully upgraded. Abort.");
			DEBUG_THROW(BasicException, SSLIT("Session has not been fully upgraded"));
		}

		try {
			const AUTO(maxRequestLength, MainConfig::getConfigFile().get<boost::uint64_t>("http_max_request_length", 16384));

			m_received.put(data, size);

			for(;;){
				if(m_state == S_UPGRADED){
					if(m_received.empty()){
						break;
					}
					LOG_POSEIDON_WARNING("Junk data received after upgrading.");
					DEBUG_THROW(BasicException, SSLIT("Junk data received after upgrading"));
				}

				boost::uint64_t sizeTotal;
				bool gotExpected;
				if(m_expectingNewLine){
					struct Helper {
						static bool traverseCallback(void *ctx, const void *data, std::size_t size){
							AUTO_REF(lfOffset, *static_cast<std::size_t *>(ctx));

							const AUTO(pos, std::memchr(data, '\n', size));
							if(!pos){
								lfOffset += size;
								return true;
							}
							lfOffset += static_cast<std::size_t>(static_cast<const char *>(pos) - static_cast<const char *>(data));
							return false;
						}
					};

					std::size_t lfOffset = 0;
					if(m_received.traverse(&Helper::traverseCallback, &lfOffset)){
						// 没找到换行符。
						sizeTotal = m_sizeTotal + m_received.size();
						gotExpected = false;
					} else {
						// 找到了。
						m_sizeExpecting = lfOffset + 1;
						sizeTotal = m_sizeTotal + m_sizeExpecting;
						gotExpected = true;
					}
				} else {
					if(m_received.size() < m_sizeExpecting){
						if(m_sizeExpecting > maxRequestLength){
							LOG_POSEIDON_WARNING("Request too large: sizeExpecting = ", m_sizeExpecting);
							DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
						}
						sizeTotal = m_sizeTotal + m_received.size();
						gotExpected = false;
					} else {
						sizeTotal = m_sizeTotal + m_sizeExpecting;
						gotExpected = true;
					}
				}
				if(sizeTotal > maxRequestLength){
					LOG_POSEIDON_WARNING("Request too large: maxRequestLength = ", maxRequestLength);
					DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
				}
				if(!gotExpected){
					break;
				}
				m_sizeTotal = sizeTotal;

				AUTO(expected, m_received.cut(m_sizeExpecting));
				if(m_expectingNewLine){
					expected.unput(); // '\n'
					if(expected.back() == '\r'){
						expected.unput();
					}
				}

				switch(m_state){
				case S_FIRST_HEADER:
					if(!expected.empty()){
						std::string line;
						expected.dump(line);

						std::size_t pos = line.find(' ');
						if(pos == std::string::npos){
							LOG_POSEIDON_WARNING("Bad HTTP header: expecting verb, line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						line[pos] = 0;
						m_header.verb = getVerbFromString(line.c_str());
						if(m_header.verb == V_INVALID_VERB){
							LOG_POSEIDON_WARNING("Bad HTTP verb: ", line.c_str());
							DEBUG_THROW(Exception, ST_NOT_IMPLEMENTED);
						}
						line.erase(0, pos + 1);

						pos = line.find(' ');
						if(pos == std::string::npos){
							LOG_POSEIDON_WARNING("Bad HTTP header: expecting URI end, line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						m_header.uri.assign(line, 0, pos);
						line.erase(0, pos + 1);

						long verEnd = 0;
						char verMajorStr[16], verMinorStr[16];
						if(std::sscanf(line.c_str(), "HTTP/%15[0-9].%15[0-9]%ln", verMajorStr, verMinorStr, &verEnd) != 2){
							LOG_POSEIDON_WARNING("Bad HTTP header: expecting HTTP version, line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						if(static_cast<unsigned long>(verEnd) != line.size()){
							LOG_POSEIDON_WARNING("Bad HTTP header: junk after HTTP version, line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						m_header.version = std::strtoul(verMajorStr, NULLPTR, 10) * 10000 + std::strtoul(verMinorStr, NULLPTR, 10);
						if((m_header.version != 10000) && (m_header.version != 10001)){
							LOG_POSEIDON_WARNING("Bad HTTP header: HTTP version not supported, verMajorStr = ", verMajorStr,
								", verMinorStr = ", verMinorStr);
							DEBUG_THROW(Exception, ST_VERSION_NOT_SUPPORTED);
						}

						pos = m_header.uri.find('?');
						if(pos != std::string::npos){
							m_header.getParams = optionalMapFromUrlEncoded(m_header.uri.substr(pos + 1));
							m_header.uri.erase(pos);
						}
						m_header.uri = urlDecode(m_header.uri);

						// m_expectingNewLine = true;
						m_state = S_HEADERS;
					} else {
						// m_state = S_FIRST_HEADER;
					}
					break;

				case S_HEADERS:
					if(!expected.empty()){
						std::string line;
						expected.dump(line);

						std::size_t pos = line.find(':');
						if(pos == std::string::npos){
							LOG_POSEIDON_WARNING("Invalid HTTP header: line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						m_header.headers.append(SharedNts(line.c_str(), pos), line.substr(line.find_first_not_of(' ', pos + 1)));

						// m_expectingNewLine = true;
						// m_state = S_HEADERS;
					} else {
						const AUTO_REF(expect, m_header.headers.get("Expect"));
						if(!expect.empty()){
							if(::strcasecmp(expect.c_str(), "100-continue") == 0){
								enqueueJob(boost::make_shared<ContinueJob>(virtualSharedFromThis<Session>()));
							} else {
								LOG_POSEIDON_WARNING("Unknown HTTP header Expect: ", expect);
								DEBUG_THROW(Exception, ST_BAD_REQUEST);
							}
						}

						const AUTO_REF(transferEncoding, m_header.headers.get("Transfer-Encoding"));
						if(transferEncoding.empty() || (::strcasecmp(transferEncoding.c_str(), "identity") == 0)){
							boost::uint64_t sizeExpecting = 0;
							const AUTO_REF(contentLength, m_header.headers.get("Content-Length"));
							if(!contentLength.empty()){
								char *endptr;
								sizeExpecting = ::strtoull(contentLength.c_str(), &endptr, 10);
								if(*endptr){
									LOG_POSEIDON_WARNING("Bad HTTP header Content-Length: ", contentLength);
									DEBUG_THROW(Exception, ST_BAD_REQUEST);
								}
							}
							m_expectingNewLine = false;
							m_sizeExpecting = sizeExpecting;
							m_state = S_IDENTITY;
						} else if(::strcasecmp(transferEncoding.c_str(), "chunked") == 0){
							// m_expectingNewLine = true;
							m_state = S_CHUNK_HEADER;
						} else {
							LOG_POSEIDON_WARNING("Unsupported Transfer-Encoding: ", transferEncoding);
							DEBUG_THROW(Exception, ST_NOT_ACCEPTABLE);
						}
					}
					break;

				case S_UPGRADED:
					std::abort();
					break;

				case S_END_OF_ENTITY:
					AuthResult authResult;
					if(m_authInfo){
						const AUTO_REF(authorization, m_header.headers.get("Authorization"));
						if(authorization.empty()){
							authResult = AUTH_REQUIRING;
						} else {
							authResult = checkAuthorization(m_authInfo, getRemoteInfo().ip.get(), m_header.verb, authorization);
						}
					} else {
						authResult = AUTH_SUCCESSFUL;
					}
					if(authResult != AUTH_SUCCESSFUL){
						enqueueJob(boost::make_shared<UnauthorizedJob>(
							virtualSharedFromThis<Session>(), authResult));

						m_header = Header();
						m_entity.clear();

						m_sizeTotal = 0;
						m_expectingNewLine = true;
						m_state = S_FIRST_HEADER;
					} else if((m_header.verb == V_CONNECT) || m_header.headers.has("Upgrade")){
						enqueueJob(boost::make_shared<UpgradeJob>(
							virtualSharedFromThis<Session>(), STD_MOVE(m_header), STD_MOVE(m_entity)));

						m_header = Header();
						m_entity.clear();

						m_sizeTotal = 0;
						m_expectingNewLine = false;
						m_sizeExpecting = (boost::uint64_t)-1;
						m_state = S_FIRST_HEADER;
					} else {
						enqueueJob(boost::make_shared<RequestJob>(
							virtualSharedFromThis<Session>(), STD_MOVE(m_header), STD_MOVE(m_entity)));

						m_header = Header();
						m_entity.clear();

						m_sizeTotal = 0;
						m_expectingNewLine = true;
						m_state = S_FIRST_HEADER;
					}
					break;

				case S_IDENTITY:
					m_entity = STD_MOVE(expected);

					m_expectingNewLine = false;
					m_sizeExpecting = 0;
					m_state = S_END_OF_ENTITY;
					break;

				case S_CHUNK_HEADER:
					if(!expected.empty()){
						std::string line;
						expected.dump(line);

						char *endptr;
						const boost::uint64_t chunkSize = ::strtoull(line.c_str(), &endptr, 16);
						if(*endptr && (*endptr != ' ')){
							LOG_POSEIDON_WARNING("Bad chunk header: ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						if(chunkSize == 0){
							m_expectingNewLine = true;
							m_state = S_CHUNKED_TRAILER;
						} else {
							m_expectingNewLine = false;
							m_sizeExpecting = chunkSize;
							m_state = S_CHUNK_DATA;
						}
					} else {
						// chunk-data 后面应该有一对 CRLF。我们在这里处理这种情况。
					}
					break;

				case S_CHUNK_DATA:
					m_entity.splice(expected);

					m_expectingNewLine = true;
					m_state = S_CHUNK_HEADER;
					break;

				case S_CHUNKED_TRAILER:
					if(!expected.empty()){
						std::string line;
						expected.dump(line);

						std::size_t pos = line.find(':');
						if(pos == std::string::npos){
							LOG_POSEIDON_WARNING("Invalid HTTP header: line = ", line);
							DEBUG_THROW(Exception, ST_BAD_REQUEST);
						}
						m_header.headers.append(SharedNts(line.c_str(), pos), line.substr(line.find_first_not_of(' ', pos + 1)));

						// m_expectingNewLine = true;
						// m_state = S_CHUNKED_TRAILER;
					} else {
						m_expectingNewLine = false;
						m_sizeExpecting = 0;
						m_state = S_END_OF_ENTITY;
					}
					break;

				default:
					LOG_POSEIDON_ERROR("Unknown state: ", static_cast<unsigned>(m_state));
					std::abort();
				}
			}
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Http::Exception thrown while parsing data, URI = ", m_header.uri,
				", status = ", static_cast<unsigned>(e.statusCode()));
			try {
				enqueueJob(boost::make_shared<ErrorJob>(
					virtualSharedFromThis<Session>(), e.statusCode(), e.headers()));
				shutdown();
			} catch(...){
				forceShutdown();
			}
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown while parsing data, URI = ", m_header.uri,
				", what = ", e.what());
			try {
				enqueueJob(boost::make_shared<ErrorJob>(
					virtualSharedFromThis<Session>(), static_cast<StatusCode>(ST_BAD_REQUEST), OptionalMap()));
				shutdown();
			} catch(...){
				forceShutdown();
			}
		}
	}
	void Session::onReadHup() NOEXCEPT {
		PROFILE_ME;

		if(m_state == S_UPGRADED){
			const AUTO(upgradedSession, getUpgradedSession());
			if(upgradedSession){
				upgradedSession->onReadHup();
			}
			return;
		}

		if((m_state == S_IDENTITY) && !m_expectingNewLine && (m_sizeExpecting == (boost::uint64_t)-1)){
			try {
				enqueueJob(boost::make_shared<RequestJob>(
					virtualSharedFromThis<Session>(), STD_MOVE(m_header), STD_MOVE(m_received)));

				m_header = Header();

				m_received.clear();
				m_expectingNewLine = true;
				m_sizeExpecting = 0;
				m_state = S_FIRST_HEADER;
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Http::Exception thrown while parsing data: URI = ", m_header.uri,
					", status = ", static_cast<unsigned>(e.statusCode()));
				try {
					enqueueJob(boost::make_shared<ErrorJob>(
						virtualSharedFromThis<Session>(), e.statusCode(), e.headers()));
					shutdown();
				} catch(...){
					forceShutdown();
				}
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown while parsing data, URI = ", m_header.uri,
					", what = ", e.what());
				try {
					enqueueJob(boost::make_shared<ErrorJob>(
						virtualSharedFromThis<Session>(), static_cast<StatusCode>(ST_BAD_REQUEST), OptionalMap()));
					shutdown();
				} catch(...){
					forceShutdown();
				}
			}
		}
	}

	boost::shared_ptr<UpgradedSessionBase> Session::onUpgrade(const Header &header, const StreamBuffer &entity){
		(void)header;
		(void)entity;

		return VAL_INIT;
	}

	boost::shared_ptr<UpgradedSessionBase> Session::getUpgradedSession() const {
		const boost::mutex::scoped_lock lock(m_upgreadedMutex);
		return m_upgradedSession;
	}

	bool Session::send(StatusCode statusCode, OptionalMap headers, StreamBuffer entity, bool fin){
		LOG_POSEIDON_DEBUG("Making HTTP response: statusCode = ", statusCode);

		StreamBuffer data;

		char first[64];
		unsigned len = (unsigned)std::sprintf(first, "HTTP/1.1 %u ", static_cast<unsigned>(statusCode));
		data.put(first, len);
		const AUTO(desc, getStatusCodeDesc(statusCode));
		data.put(desc.descShort);
		data.put("\r\n");

		if(!entity.empty()){
			AUTO_REF(contentType, headers.create("Content-Type")->second);
			if(contentType.empty()){
				contentType.assign("text/plain; charset=utf-8");
			}
		}
		headers.set("Content-Length", boost::lexical_cast<std::string>(entity.size()));
		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			if(it->second.empty()){
				continue;
			}
			data.put(it->first.get());
			data.put(": ");
			data.put(it->second.data(), it->second.size());
			data.put("\r\n");
		}
		data.put("\r\n");

		data.splice(entity);
		return TcpSessionBase::send(STD_MOVE(data), fin);
	}
	bool Session::sendDefault(StatusCode statusCode, OptionalMap headers, bool fin){
		LOG_POSEIDON_DEBUG("Making default HTTP response: statusCode = ", statusCode, ", fin = ", fin);

		StreamBuffer entity;
		if(static_cast<unsigned>(statusCode) / 100 >= 4){
			headers.set("Content-Type", "text/html; charset=utf-8");

			entity.put("<html><head><title>");
			const AUTO(desc, getStatusCodeDesc(statusCode));
			entity.put(desc.descShort);
			entity.put("</title></head><body><h1>");
			entity.put(desc.descShort);
			entity.put("</h1><hr /><p>");
			entity.put(desc.descLong);
			entity.put("</p></body></html>");
		}
		return send(statusCode, STD_MOVE(headers), STD_MOVE(entity), fin);
	}
}

}
