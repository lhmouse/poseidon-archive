// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "system_session.hpp"
#include "singletons/system_server.hpp"
#include "singletons/main_config.hpp"
#include "system_servlet_base.hpp"
#include "profiler.hpp"
#include "log.hpp"
#include "json.hpp"
#include "http/authentication.hpp"
#include "http/urlencoded.hpp"
#include "http/exception.hpp"

namespace Poseidon {

SystemSession::SystemSession(Move<UniqueFile> socket, boost::shared_ptr<const Http::AuthenticationContext> auth_ctx)
	: Http::Session(STD_MOVE(socket))
	, m_auth_ctx(STD_MOVE(auth_ctx))
	, m_initialized(false), m_decoded_uri(), m_servlet()
{
	LOG_POSEIDON_INFO("SystemSession constructor: remote = ", get_remote_info());
}
SystemSession::~SystemSession(){
	LOG_POSEIDON_INFO("SystemSession destructor: remote = ", get_remote_info());
}

void SystemSession::initialize_once(const Http::RequestHeaders &request_headers){
	PROFILE_ME;

	if(m_initialized){
		return;
	}

	const AUTO(user, Http::check_authentication_simple(m_auth_ctx, false, get_remote_info(), request_headers));
	LOG_POSEIDON_INFO("SystemSession authentication succeeded: remote = ", get_remote_info(), ", user = ", user, ", URI = ", request_headers.uri, ", headers = ", request_headers.headers);

	Buffer_istream bis;
	bis.set_buffer(StreamBuffer(request_headers.uri));
	Http::url_decode(bis, m_decoded_uri);
	if(m_decoded_uri.empty()){
		DEBUG_THROW(Http::Exception, Http::ST_BAD_REQUEST);
	}
	if(*m_decoded_uri.begin() != '/'){
		DEBUG_THROW(Http::Exception, Http::ST_BAD_REQUEST);
	}
	LOG_POSEIDON_DEBUG("Decoded request URI: ", m_decoded_uri);

	switch(request_headers.verb){
	case Http::V_GET:
	case Http::V_HEAD:
	case Http::V_POST: {
		AUTO(servlet, SystemServer::get_servlet(m_decoded_uri.c_str()));
		if(!servlet){
			LOG_POSEIDON_WARNING("SystemSession URI not handled: ", m_decoded_uri);
			DEBUG_THROW(Http::Exception, Http::ST_NOT_FOUND);
		}
		m_servlet = STD_MOVE(servlet);
		break; }

	case Http::V_OPTIONS: {
		LOG_POSEIDON_DEBUG("Query options: request_headers.headers");
		break; }

	default:
		DEBUG_THROW(Http::Exception, Http::ST_METHOD_NOT_ALLOWED);
	}

	++m_initialized;
}

void SystemSession::on_sync_expect(Http::RequestHeaders request_headers){
	PROFILE_ME;

	initialize_once(request_headers);

	Http::Session::on_sync_expect(STD_MOVE(request_headers));
}
void SystemSession::on_sync_request(Http::RequestHeaders request_headers, StreamBuffer request_entity){
	PROFILE_ME;

	initialize_once(request_headers);

	const bool keep_alive = Http::is_keep_alive_enabled(request_headers);

	Http::ResponseHeaders response_headers;
	StreamBuffer response_entity;

	switch(request_headers.verb){
	case Http::V_GET:
	case Http::V_HEAD: {
		DEBUG_THROW_ASSERT(m_servlet);

		try {
			JsonObject response;
			response = m_servlet->handle_http_get();
			LOG_POSEIDON_DEBUG("SystemSession GET response: ", response);

			response_headers.status_code = Http::ST_OK;
			response_headers.headers.set(sslit("Content-Type"), "application/json");
			if(request_headers.verb != Http::V_HEAD){
				Buffer_ostream bos;
				response.dump(bos);
				response_entity = STD_MOVE(bos.get_buffer());
			}
		} catch(Http::Exception &e){
			LOG_POSEIDON_WARNING("Http::Exception thrown: status_code = ", e.get_status_code(), ", headers = ", e.get_headers());
			response_headers.status_code = e.get_status_code();
			response_headers.headers = e.get_headers();
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			response_headers.status_code = Http::ST_INTERNAL_SERVER_ERROR;
		}
		break; }

	case Http::V_POST: {
		DEBUG_THROW_ASSERT(m_servlet);

		try {
			Buffer_istream bis;
			bis.set_buffer(STD_MOVE(request_entity));
			JsonObject request;
			request.parse(bis);
			if(!bis){
				LOG_POSEIDON_WARNING("Invalid JSON request before ", bis.get_buffer());
				DEBUG_THROW(Http::Exception, Http::ST_BAD_REQUEST);
			}
			LOG_POSEIDON_DEBUG("SystemSession POST request: ", request);

			JsonObject response;
			response = m_servlet->handle_http_post(STD_MOVE(request));
			LOG_POSEIDON_DEBUG("SystemSession POST response: ", response);

			response_headers.status_code = Http::ST_OK;
			response_headers.headers.set(sslit("Content-Type"), "application/json");
			Buffer_ostream bos;
			response.dump(bos);
			response_entity = STD_MOVE(bos.get_buffer());
		} catch(Http::Exception &e){
			LOG_POSEIDON_WARNING("Http::Exception thrown: status_code = ", e.get_status_code(), ", headers = ", e.get_headers());
			response_headers.status_code = e.get_status_code();
			response_headers.headers = e.get_headers();
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			response_headers.status_code = Http::ST_INTERNAL_SERVER_ERROR;
		}
		break; }

	case Http::V_OPTIONS: {
		response_headers.status_code = Http::ST_OK;
		break; }

	default:
		DEBUG_THROW(Http::Exception, Http::ST_METHOD_NOT_ALLOWED);
	}

	response_headers.version = 10001;
	response_headers.reason = Http::get_status_code_desc(response_headers.status_code).desc_short;
	response_headers.headers.set(sslit("Connection"), keep_alive ? "Keep-Alive" : "Close");
	response_headers.headers.set(sslit("Access-Control-Allow-Origin"), "*");
	response_headers.headers.set(sslit("Access-Control-Allow-Headers"), "Authorization, Content-Type");
	response_headers.headers.set(sslit("Access-Control-Allow-Methods"), "GET, HEAD, OPTIONS");

	Http::Session::send(STD_MOVE(response_headers), STD_MOVE(response_entity));
}

}
