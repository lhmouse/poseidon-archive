// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "simple_http_client_daemon.hpp"
#include "main_config.hpp"
#include "dns_daemon.hpp"
#include "job_dispatcher.hpp"
#include "epoll_daemon.hpp"
#include "../log.hpp"
#include "../atomic.hpp"
#include "../profiler.hpp"
#include "../job_base.hpp"
#include "../flags.hpp"
#include "../system_exception.hpp"
#include "../errno.hpp"
#include "../checked_arithmetic.hpp"
#include "../http/exception.hpp"
#include "../http/low_level_client.hpp"
#include "../http/urlencoded.hpp"
#include <poll.h>
#include <sys/types.h>
#include <sys/socket.h>

namespace Poseidon {

template class Promise_container<Simple_http_response>;

namespace {
	bool can_be_redirected(const Simple_http_request &request){
		switch(request.request_headers.verb){
		case Http::verb_head:
			return false;
		case Http::verb_get:
			return !request.dont_redirect_get;
		default:
			return request.redirect_non_get;
		}
	}

	bool check_redirect(Simple_http_request &request, const Http::Response_headers &respones_headers){
		PROFILE_ME;

		const AUTO_REF(location, respones_headers.headers.get("Location"));
		if(location.empty()){
			return false;
		}
		switch(respones_headers.status_code){
		case Http::status_moved_permanently: // 301
		case Http::status_found: // 302
		case Http::status_temporary_redirect: // 307
			LOG_POSEIDON_DEBUG("Redirecting intactly: location = ", location);
			request.request_headers.uri = location;
			return true;
		case Http::status_see_other: // 303
			LOG_POSEIDON_DEBUG("Redirecting intactly: location = ", location);
			request.request_headers.verb = Http::verb_get;
			request.request_headers.uri = location;
			request.request_entity.clear();
			return true;
		default:
			LOG_POSEIDON_DEBUG("This is a final response: status_code = ", respones_headers.status_code);
			return false;
		}
	}

	struct Simple_http_client_params {
		std::string host;
		boost::uint16_t port;
		bool use_ssl;
		Http::Request_headers request_headers;
	};

	Simple_http_client_params parse_simple_http_client_params(Http::Request_headers request_headers){
		PROFILE_ME;
		DEBUG_THROW_UNLESS(!request_headers.uri.empty(), Exception, sslit("Request URI is empty"));
		DEBUG_THROW_UNLESS(request_headers.uri.at(0) != '/', Exception, sslit("Relative request URI is not allowed"));

		Simple_http_client_params params = { std::string(), 80, false };
		// uri = "http://www.example.com:80/foo/bar/page.html?param=value"
		AUTO(pos, request_headers.uri.find("://"));
		if(pos != std::string::npos){
			request_headers.uri.at(pos) = 0;
			LOG_POSEIDON_TRACE("Request protocol = ", request_headers.uri.c_str());
			if(::strcasecmp(request_headers.uri.c_str(), "http") == 0){
				params.port = 80;
				params.use_ssl = false;
			} else if(::strcasecmp(request_headers.uri.c_str(), "https") == 0){
				params.port = 443;
				params.use_ssl = true;
			} else {
				LOG_POSEIDON_WARNING("Unsupported protocol: ", request_headers.uri.c_str());
				DEBUG_THROW(Exception, sslit("Unsupported protocol"));
			}
			request_headers.uri.erase(0, pos + 3);
		}
		// uri = "www.example.com:80/foo/bar/page.html?param=value"
		pos = request_headers.uri.find('/');
		if(pos != std::string::npos){
			params.host = request_headers.uri.substr(0, pos);
			request_headers.uri.erase(0, pos);
		} else {
			params.host = STD_MOVE(request_headers.uri);
			request_headers.uri = "/";
		}
		// host = "www.example.com:80"
		// uri = "/foo/bar/page.html?param=value"
		if(params.host.at(0) == '['){
			pos = params.host.find(']');
			DEBUG_THROW_UNLESS(pos != std::string::npos, Exception, sslit("Invalid IPv6 address"));
			pos = params.host.find(':', pos + 1);
		} else {
			pos = params.host.find(':');
		}
		if(pos != std::string::npos){
			char *eptr;
			const unsigned long port_val = std::strtoul(params.host.c_str() + pos + 1, &eptr, 10);
			DEBUG_THROW_UNLESS(*eptr == 0, Exception, sslit("Invalid port string"));
			DEBUG_THROW_UNLESS((1 <= port_val) && (port_val <= 65534), Exception, sslit("Invalid port number"));
			params.port = boost::numeric_cast<boost::uint16_t>(port_val);
			params.host.erase(pos);
		}
		// host = "www.example.com"
		// port = 80
		// uri = "/foo/bar/page.html?param=value"
		pos = request_headers.uri.find('?');
		if(pos != std::string::npos){
			Optional_map temp_params;
			Buffer_istream bis;
			bis.set_buffer(Stream_buffer(request_headers.uri.c_str() + pos + 1, request_headers.uri.size() - pos - 1));
			Http::url_decode_params(bis, temp_params);
			if(bis && !temp_params.empty()){
				for(AUTO(it, request_headers.get_params.begin()); it != request_headers.get_params.end(); ++it){
					temp_params.append(it->first, STD_MOVE(it->second));
				}
				request_headers.get_params.swap(temp_params);
			}
			request_headers.uri.erase(pos);
		}
		// host = "www.example.com"
		// port = 80
		// uri = "/foo/bar/page.html"
		// get_params = "{ param=value }"
		params.request_headers = STD_MOVE(request_headers);
		params.request_headers.version = 10001;
		params.request_headers.headers.set(sslit("Host"), params.host);
		return params;
	}

	void poll_internal(const boost::shared_ptr<Socket_base> &socket){
		PROFILE_ME;

		bool readable = false, writeable = false;
		unsigned char buffer[2048];
		do {
			::pollfd pset = { socket->get_fd(), POLLIN | POLLOUT };
			int err_code = ::poll(&pset, 1, -1);
			if(err_code < 0){
				err_code = errno;
				if(err_code != EINTR){
					LOG_POSEIDON_ERROR("::poll() failed! errno was ", err_code, " (", get_error_desc(err_code), ")");
					DEBUG_THROW(System_exception, err_code);
				}
				continue;
			}
			if(has_any_flags_of(pset.revents, POLLIN) && has_none_flags_of(pset.revents, POLLERR)){
				readable = true;
				err_code = socket->poll_read_and_process(buffer, sizeof(buffer), readable);
				LOG_POSEIDON_TRACE("Socket read result: socket = ", socket, ", typeid = ", typeid(*socket).name(), ", err_code = ", err_code);
				if((err_code != 0) && (err_code != EINTR) && (err_code != EWOULDBLOCK) && (err_code != EAGAIN)){
					LOG_POSEIDON_DEBUG("Socket read error: err_code = ", err_code, " (", get_error_desc(err_code), ")");
					socket->force_shutdown();
				}
			}
			if(has_any_flags_of(pset.revents, POLLOUT) && has_none_flags_of(pset.revents, POLLERR)){
				writeable = true;
				Mutex::Unique_lock write_lock;
				err_code = socket->poll_write(write_lock, buffer, sizeof(buffer), writeable);
				LOG_POSEIDON_TRACE("Socket write result: socket = ", socket, ", typeid = ", typeid(*socket).name(), ", err_code = ", err_code);
				if((err_code != 0) && (err_code != EINTR) && (err_code != EWOULDBLOCK) && (err_code != EAGAIN)){
					LOG_POSEIDON_DEBUG("Socket write error: err_code = ", err_code, " (", get_error_desc(err_code), ")");
					socket->force_shutdown();
				}
			}
			if(has_any_flags_of(pset.revents, POLLHUP | POLLERR)){
				if(socket->did_time_out()){
					err_code = ETIMEDOUT;
				} else if(has_any_flags_of(pset.revents, POLLERR)){
					::socklen_t err_len = sizeof(err_code);
					if(::getsockopt(socket->get_fd(), SOL_SOCKET, SO_ERROR, &err_code, &err_len) != 0){
						err_code = errno;
						LOG_POSEIDON_WARNING("::getsockopt() failed: fd = ", socket->get_fd(), ", err_code = ", err_code, " (", get_error_desc(err_code), ")");
					}
				} else {
					err_code = 0;
				}
				socket->mark_shutdown();
				socket->on_close(err_code);
			}
		} while(!socket->has_been_shutdown_read());
	}

	class Simple_http_client : public Http::Low_level_client, public Promise {
	private:
		Http::Response_headers m_response_headers;
		Stream_buffer m_response_entity;
		bool m_finished;

	public:
		Simple_http_client(const Sock_addr &sock_addr, bool use_ssl)
			: Http::Low_level_client(sock_addr, use_ssl)
			, m_response_headers(), m_response_entity(), m_finished(false)
		{
			//
		}

	protected:
		void on_close(int err_code) OVERRIDE {
			Promise::set_success(false);
			return Http::Low_level_client::on_close(err_code);
		}

		void on_low_level_response_headers(Http::Response_headers response_headers, boost::uint64_t /*content_length*/) OVERRIDE {
			m_response_headers = STD_MOVE(response_headers);
			m_finished = true;
		}
		void on_low_level_response_entity(boost::uint64_t /*entity_offset*/, Stream_buffer entity) OVERRIDE {
			m_response_entity.splice(entity);
		}
		boost::shared_ptr<Http::Upgraded_session_base> on_low_level_response_end(boost::uint64_t /*content_length*/, Optional_map /*headers*/) OVERRIDE {
			shutdown_read();
			shutdown_write();
			return VAL_INIT;
		}

	public:
		bool send(Http::Request_headers request_headers, Stream_buffer request_entity){
			request_headers.headers.set(sslit("Connection"), "Close");
			request_headers.headers.erase("Expect");
			DEBUG_THROW_UNLESS(Http::Low_level_client::send(STD_MOVE(request_headers), STD_MOVE(request_entity)), Exception, sslit("Failed to send data to remote server"));
			return true;
		}

		// WARNING: Only call these functions after `has_been_shutdown_read()` returns `true` to avoid race conditions!
		Http::Response_headers &get_response_headers(){
			return m_response_headers;
		}
		Stream_buffer &get_response_entity(){
			return m_response_entity;
		}
		bool is_finished() const {
			return m_finished;
		}
	};

	class Async_perform_job : public Job_base {
	private:
		boost::weak_ptr<Promise_container<Simple_http_response> > m_weak_promise;
		Simple_http_request m_request;

	public:
		Async_perform_job(const boost::shared_ptr<Promise_container<Simple_http_response> > &promise, Simple_http_request request)
			: m_weak_promise(promise), m_request(STD_MOVE(request))
		{
			//
		}

	protected:
		boost::weak_ptr<const void> get_category() const FINAL {
			return VAL_INIT;
		}
		void perform() FINAL {
			PROFILE_ME;

			AUTO_REF(request, m_request);
			Simple_http_response response;
			STD_EXCEPTION_PTR except;
			try {
				boost::shared_ptr<Simple_http_client> client;

				const bool should_check_redirect = can_be_redirected(request);
				const AUTO(max_redirect_count, Main_config::get<std::size_t>("simple_http_client_max_redirect_count", 10));
				std::size_t retry_count_remaining = checked_add<std::size_t>(max_redirect_count, 1);
				do {
					const AUTO(verb, request.request_headers.verb);
					LOG_POSEIDON_DEBUG("Trying: ", Http::get_string_from_verb(verb), " ", request.request_headers.uri);
					AUTO(params, parse_simple_http_client_params(should_check_redirect ? request.request_headers : STD_MOVE_IDN(request.request_headers)));
					const AUTO(promised_sock_addr, Dns_daemon::enqueue_for_looking_up(params.host, params.port));
					Job_dispatcher::yield(promised_sock_addr, true);
					const AUTO_REF(sock_addr, promised_sock_addr->get());
					client = boost::make_shared<Simple_http_client>(sock_addr, params.use_ssl);
					client->set_no_delay(true);
					client->send(STD_MOVE(params.request_headers), should_check_redirect ? request.request_entity : STD_MOVE_IDN(request.request_entity));
					Epoll_daemon::add_socket(client);
					Job_dispatcher::yield(client, true);
					DEBUG_THROW_UNLESS(client->is_finished() || (verb == Http::verb_head), Exception, sslit("Connection was closed prematurely"));
				} while(should_check_redirect && (--retry_count_remaining != 0) && check_redirect(request, client->get_response_headers()));

				Simple_http_response temp = { STD_MOVE(client->get_response_headers()), STD_MOVE(client->get_response_entity()) };
				response = STD_MOVE(temp);
			} catch(std::exception &e){
				LOG_POSEIDON_DEBUG("std::exception thrown: what = ", e.what());
				except = STD_CURRENT_EXCEPTION();
			} catch(...){
				LOG_POSEIDON_DEBUG("Unknown exception thrown.");
				except = STD_CURRENT_EXCEPTION();
			}
			const AUTO(promise, m_weak_promise.lock());
			if(promise){
				if(except){
					promise->set_exception(STD_MOVE(except), false);
				} else {
					promise->set_success(STD_MOVE(response), false);
				}
			}
		}
	};

	volatile bool g_running = false;
}

void Simple_http_client_daemon::start(){
	if(atomic_exchange(g_running, true, memory_order_acq_rel) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::terminate();
	}
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Starting simple HTTP client daemon...");

	//
}
void Simple_http_client_daemon::stop(){
	if(atomic_exchange(g_running, false, memory_order_acq_rel) == false){
		return;
	}
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Stopping simple HTTP client daemon...");

	//
}

Simple_http_response Simple_http_client_daemon::perform(Simple_http_request request){
	PROFILE_ME;

	boost::shared_ptr<Simple_http_client> client;

	const bool should_check_redirect = can_be_redirected(request);
	const AUTO(max_redirect_count, Main_config::get<std::size_t>("simple_http_client_max_redirect_count", 10));
	std::size_t retry_count_remaining = checked_add<std::size_t>(max_redirect_count, 1);
	do {
		const AUTO(verb, request.request_headers.verb);
		LOG_POSEIDON_DEBUG("Trying: ", Http::get_string_from_verb(verb), " ", request.request_headers.uri);
		AUTO(params, parse_simple_http_client_params(should_check_redirect ? request.request_headers : STD_MOVE_IDN(request.request_headers)));
		const AUTO(sock_addr, Dns_daemon::look_up(params.host, params.port));
		client = boost::make_shared<Simple_http_client>(sock_addr, params.use_ssl);
		client->set_no_delay(true);
		client->send(STD_MOVE(params.request_headers), should_check_redirect ? request.request_entity : STD_MOVE_IDN(request.request_entity));
		poll_internal(client);
		DEBUG_THROW_UNLESS(client->is_finished() || (verb == Http::verb_head), Exception, sslit("Connection was closed prematurely"));
	} while(should_check_redirect && (--retry_count_remaining != 0) && check_redirect(request, client->get_response_headers()));

	Simple_http_response response = { STD_MOVE(client->get_response_headers()), STD_MOVE(client->get_response_entity()) };
	return response;
}

boost::shared_ptr<const Promise_container<Simple_http_response> > Simple_http_client_daemon::enqueue_for_performing(Simple_http_request request){
	PROFILE_ME;

	AUTO(promise, boost::make_shared<Promise_container<Simple_http_response> >());
	Job_dispatcher::enqueue(boost::make_shared<Async_perform_job>(promise, STD_MOVE(request)), VAL_INIT);
	return STD_MOVE_IDN(promise);
}

}
