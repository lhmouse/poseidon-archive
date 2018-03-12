// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "system_http_session.hpp"
#include "singletons/system_http_server.hpp"
#include "singletons/main_config.hpp"
#include "system_http_servlet_base.hpp"
#include "profiler.hpp"
#include "log.hpp"
#include "json.hpp"
#include "http/authentication.hpp"
#include "http/urlencoded.hpp"
#include "http/exception.hpp"

namespace Poseidon {

SystemHttpSession::SystemHttpSession(Move<UniqueFile> socket, boost::shared_ptr<const Http::AuthenticationContext> auth_ctx)
	: Http::Session(STD_MOVE(socket))
	, m_auth_ctx(STD_MOVE(auth_ctx))
	, m_initialized(false), m_decoded_uri(), m_servlet()
{
	LOG_POSEIDON_INFO("SystemHttpSession constructor: remote = ", get_remote_info());
}
SystemHttpSession::~SystemHttpSession(){
	LOG_POSEIDON_INFO("SystemHttpSession destructor: remote = ", get_remote_info());
}

void SystemHttpSession::initialize_once(const Http::RequestHeaders &request_headers){
	PROFILE_ME;

	if(m_initialized){
		return;
	}

	const AUTO(user, Http::check_authentication_simple(m_auth_ctx, false, get_remote_info(), request_headers));
	LOG_POSEIDON_INFO("SystemHttpSession authentication succeeded: remote = ", get_remote_info(), ", user = ", user, ", URI = ", request_headers.uri, ", headers = ", request_headers.headers);

	Buffer_istream bis;
	bis.set_buffer(StreamBuffer(request_headers.uri));
	Http::url_decode(bis, m_decoded_uri);
	DEBUG_THROW_UNLESS(!m_decoded_uri.empty(), Http::Exception, Http::ST_BAD_REQUEST);
	DEBUG_THROW_UNLESS(m_decoded_uri[0] == '/', Http::Exception, Http::ST_BAD_REQUEST);
	LOG_POSEIDON_DEBUG("Decoded request URI: ", m_decoded_uri);

	switch(request_headers.verb){
	case Http::V_OPTIONS:
		break;
	case Http::V_GET:
	case Http::V_HEAD:
	case Http::V_POST:
		for(;;){
			m_servlet = SystemHttpServer::get_servlet(m_decoded_uri.c_str());
			if(m_servlet){
				break;
			}
			if(*m_decoded_uri.rbegin() == '/'){
				LOG_POSEIDON_WARNING("SystemHttpSession URI not handled: ", m_decoded_uri);
				DEBUG_THROW(Http::Exception, Http::ST_NOT_FOUND);
			}
			m_decoded_uri.push_back('/');
			LOG_POSEIDON_DEBUG("Retrying: ", m_decoded_uri);
		}
		break;
	default:
		DEBUG_THROW(Http::Exception, Http::ST_METHOD_NOT_ALLOWED);
	}

	++m_initialized;
}

void SystemHttpSession::on_sync_expect(Http::RequestHeaders request_headers){
	PROFILE_ME;

	initialize_once(request_headers);

	Http::Session::on_sync_expect(STD_MOVE(request_headers));
}
void SystemHttpSession::on_sync_request(Http::RequestHeaders request_headers, StreamBuffer request_entity){
	PROFILE_ME;

	initialize_once(request_headers);

	const bool keep_alive = Http::is_keep_alive_enabled(request_headers);

	Http::ResponseHeaders response_headers;
	response_headers.version = 10001;
	response_headers.status_code = Http::ST_OK;
	response_headers.reason = "OK";
	response_headers.headers.set(sslit("Connection"), keep_alive ? "Keep-Alive" : "Close");
	response_headers.headers.set(sslit("Access-Control-Allow-Origin"), "*");
	response_headers.headers.set(sslit("Access-Control-Allow-Headers"), "Authorization, Content-Type");
	response_headers.headers.set(sslit("Access-Control-Allow-Methods"), "OPTIONS, GET, HEAD, POST");

	JsonObject request;
	Buffer_istream bis;
	JsonObject response;
	Buffer_ostream bos;

	switch(request_headers.verb){
	case Http::V_OPTIONS:
		Http::Session::send(STD_MOVE(response_headers));
		break;
	case Http::V_GET:
	case Http::V_HEAD:
	case Http::V_POST:
		DEBUG_THROW_ASSERT(m_servlet);
		if(request_headers.verb != Http::V_POST){
			// no parameters
		} else {
			LOG_POSEIDON_DEBUG("Parsing POST entity as JSON Object: ", request_entity);
			bis.set_buffer(STD_MOVE(request_entity));
			request.parse(bis);
			DEBUG_THROW_UNLESS(bis, Http::Exception, Http::ST_BAD_REQUEST);
		}
		LOG_POSEIDON_DEBUG("SystemHttpSession request: ", request);
		if(request_headers.verb != Http::V_POST){
			m_servlet->handle_get(response);
		} else {
			m_servlet->handle_post(response, STD_MOVE(request));
		}
		LOG_POSEIDON_DEBUG("SystemHttpSession response: ", response);
		response.dump(bos);
		response_headers.headers.set(sslit("Content-Type"), "application/json");
		Http::Session::send_chunked_header(STD_MOVE(response_headers));
		if(request_headers.verb == Http::V_HEAD){
			LOG_POSEIDON_DEBUG("The response entity for a HEAD request will be discarded.");
			break;
		}
		Http::Session::send_chunk(STD_MOVE(bos.get_buffer()));
		Http::Session::send_chunked_trailer();
		break;
	default:
		DEBUG_THROW(Http::Exception, Http::ST_METHOD_NOT_ALLOWED);
	}
}

}
