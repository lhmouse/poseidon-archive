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

System_http_session::System_http_session(Move<Unique_file> socket, boost::shared_ptr<const Http::Authentication_context> auth_ctx)
	: Http::Session(STD_MOVE(socket))
	, m_auth_ctx(STD_MOVE(auth_ctx))
	, m_initialized(false), m_decoded_uri(), m_servlet()
{
	LOG_POSEIDON_INFO("System_http_session constructor: remote = ", get_remote_info());
}
System_http_session::~System_http_session(){
	LOG_POSEIDON_INFO("System_http_session destructor: remote = ", get_remote_info());
}

void System_http_session::initialize_once(const Http::Request_headers &request_headers){
	PROFILE_ME;

	if(m_initialized){
		return;
	}

	const AUTO(user, Http::check_authentication_simple(m_auth_ctx, false, get_remote_info(), request_headers));
	LOG_POSEIDON_INFO("System_http_session authentication succeeded: remote = ", get_remote_info(), ", user = ", user, ", URI = ", request_headers.uri, ", headers = ", request_headers.headers);

	Buffer_istream bis;
	bis.set_buffer(Stream_buffer(request_headers.uri));
	Http::url_decode(bis, m_decoded_uri);
	DEBUG_THROW_UNLESS(!m_decoded_uri.empty(), Http::Exception, Http::status_bad_request);
	DEBUG_THROW_UNLESS(m_decoded_uri[0] == '/', Http::Exception, Http::status_bad_request);
	LOG_POSEIDON_DEBUG("Decoded request URI: ", m_decoded_uri);

	switch(request_headers.verb){
	case Http::verb_options:
		break;
	case Http::verb_get:
	case Http::verb_head:
	case Http::verb_post:
		for(;;){
			m_servlet = System_http_server::get_servlet(m_decoded_uri.c_str());
			if(m_servlet){
				break;
			}
			if(*m_decoded_uri.rbegin() == '/'){
				LOG_POSEIDON_WARNING("System_http_session URI not handled: ", m_decoded_uri);
				DEBUG_THROW(Http::Exception, Http::status_not_found);
			}
			m_decoded_uri.push_back('/');
			LOG_POSEIDON_DEBUG("Retrying: ", m_decoded_uri);
		}
		break;
	default:
		DEBUG_THROW(Http::Exception, Http::status_method_not_allowed);
	}

	m_initialized = true;
}

void System_http_session::on_sync_expect(Http::Request_headers request_headers){
	PROFILE_ME;

	initialize_once(request_headers);

	Http::Session::on_sync_expect(STD_MOVE(request_headers));
}
void System_http_session::on_sync_request(Http::Request_headers request_headers, Stream_buffer request_entity){
	PROFILE_ME;

	initialize_once(request_headers);

	const bool keep_alive = Http::is_keep_alive_enabled(request_headers);

	Http::Response_headers response_headers;
	response_headers.version = 10001;
	response_headers.status_code = Http::status_ok;
	response_headers.reason = "OK";
	response_headers.headers.set(Rcnts::view("Connection"), keep_alive ? "Keep-Alive" : "Close");
	response_headers.headers.set(Rcnts::view("Access-Control-Allow-Origin"), "*");
	response_headers.headers.set(Rcnts::view("Access-Control-Allow-Headers"), "Authorization, Content-Type");
	response_headers.headers.set(Rcnts::view("Access-Control-Allow-Methods"), "OPTIONS, GET, HEAD, POST");

	Json_object request;
	Buffer_istream bis;
	Json_object response;
	Buffer_ostream bos;

	switch(request_headers.verb){
	case Http::verb_options:
		Http::Session::send(STD_MOVE(response_headers));
		break;
	case Http::verb_get:
	case Http::verb_head:
	case Http::verb_post:
		DEBUG_THROW_ASSERT(m_servlet);
		if(request_headers.verb != Http::verb_post){
			// no parameters
		} else {
			LOG_POSEIDON_DEBUG("Parsing POST entity as JSON Object: ", request_entity);
			bis.set_buffer(STD_MOVE(request_entity));
			request.parse(bis);
			DEBUG_THROW_UNLESS(bis, Http::Exception, Http::status_bad_request);
		}
		LOG_POSEIDON_DEBUG("System_http_session request: ", request);
		if(request_headers.verb != Http::verb_post){
			m_servlet->handle_get(response);
		} else {
			m_servlet->handle_post(response, STD_MOVE(request));
		}
		LOG_POSEIDON_DEBUG("System_http_session response: ", response);
		response.dump(bos);
		response_headers.headers.set(Rcnts::view("Content-Type"), "application/json");
		Http::Session::send_chunked_header(STD_MOVE(response_headers));
		if(request_headers.verb == Http::verb_head){
			LOG_POSEIDON_DEBUG("The response entity for a HEAD request will be discarded.");
			break;
		}
		Http::Session::send_chunk(STD_MOVE(bos.get_buffer()));
		Http::Session::send_chunked_trailer();
		break;
	default:
		DEBUG_THROW(Http::Exception, Http::status_method_not_allowed);
	}
}

}
