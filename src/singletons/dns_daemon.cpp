// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "dns_daemon.hpp"
#include <netdb.h>
#include <unistd.h>
#include "../log.hpp"
#include "../profiler.hpp"
#include "../exception.hpp"
#include "../atomic.hpp"
#include "../job_promise.hpp"
#include "../sock_addr.hpp"
#include "../ip_port.hpp"

namespace Poseidon {

namespace {
	volatile bool g_running = false;
	volatile std::size_t g_pending_callback_count = 0;

	struct DnsCallbackParam : NONCOPYABLE {
		const boost::shared_ptr<SockAddr> sock_addr;

		const std::string host;
		const unsigned port;
		const boost::shared_ptr<JobPromise> promise;

		std::string host_str;
		char port_str[16];
		::gaicb cb;

		DnsCallbackParam(boost::shared_ptr<SockAddr> sock_addr_,
			std::string host_, unsigned port_, boost::shared_ptr<JobPromise> promise_)
			: sock_addr(STD_MOVE(sock_addr_))
			, host(STD_MOVE(host_)), port(port_), promise(STD_MOVE(promise_))
		{
			assert(!host.empty());

			if((host.begin()[0] == '[') && (host.end()[-1] == ']')){
				host_str.assign(host.begin() + 1, host.end() - 1);
			} else {
				host_str = host;
			}
			std::sprintf(port_str, "%u", port);

			cb.ar_name    = host_str.c_str();
			cb.ar_service = port_str;
			cb.ar_request = NULLPTR;
			cb.ar_result  = NULLPTR;
		}
		~DnsCallbackParam(){
			if(cb.ar_result){
				::freeaddrinfo(cb.ar_result);
			}
		}
	};

	void dns_callback(::sigval sigval_param) NOEXCEPT {
		PROFILE_ME;

		Logger::set_thread_tag("   D"); // DNS
		const boost::scoped_ptr<DnsCallbackParam> param(static_cast<DnsCallbackParam *>(sigval_param.sival_ptr));

		try {
			const int gai_code = ::gai_error(&(param->cb));
			const char *err_msg = "";
			if(gai_code != 0){
				err_msg = ::gai_strerror(gai_code);
				LOG_POSEIDON_DEBUG("DNS lookup failure: host = ", param->host, ", gai_code = ", gai_code, ", err_msg = ", err_msg);
				DEBUG_THROW(Exception, SharedNts(err_msg));
			}
			*(param->sock_addr) = SockAddr(param->cb.ar_result->ai_addr, param->cb.ar_result->ai_addrlen);
			LOG_POSEIDON_DEBUG("DNS lookup success: host:port = ", param->host, ':', param->port,
				", result = ", get_ip_port_from_sock_addr(*(param->sock_addr)));
			param->promise->set_success();
		} catch(Exception &e){
			LOG_POSEIDON_INFO("Exception thrown in DNS loop: what = ", e.what());
			param->promise->set_exception(boost::copy_exception(e));
		} catch(std::exception &e){
			LOG_POSEIDON_INFO("std::exception thrown in DNS loop: what = ", e.what());
			try {
				param->promise->set_exception(boost::copy_exception(std::runtime_error(e.what())));
			} catch(...){
				param->promise->set_exception(boost::current_exception());
			}
		} catch(...){
			LOG_POSEIDON_INFO("Unknown exception thrown in DNS loop.");
			param->promise->set_exception(boost::current_exception());
		}

		atomic_sub(g_pending_callback_count, 1, ATOMIC_RELAXED);
	}
}

void DnsDaemon::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting DNS daemon...");

	// 无事可做。
}
void DnsDaemon::stop(){
	if(atomic_exchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping DNS daemon...");

	for(;;){
		const AUTO(count, atomic_load(g_pending_callback_count, ATOMIC_RELAXED));
		if(count == 0){
			break;
		}
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for ", count, " pending DNS callbacks...");
		::gai_cancel(NULLPTR);
		::usleep(100000);
	}
}

SockAddr DnsDaemon::sync_lookup(std::string host, unsigned port){
	PROFILE_ME;

	DnsCallbackParam param(VAL_INIT, STD_MOVE(host), port, VAL_INIT);
	const int gai_code = ::getaddrinfo(param.cb.ar_name, param.cb.ar_service, param.cb.ar_request, &param.cb.ar_result);
	if(gai_code != 0){
		LOG_POSEIDON_WARNING("DNS failure: gai_code = ", gai_code, ", err_msg = ", ::gai_strerror(gai_code));
		DEBUG_THROW(Exception, sslit("DNS failure"));
	}
	return SockAddr(param.cb.ar_result->ai_addr, param.cb.ar_result->ai_addrlen);
}

boost::shared_ptr<const JobPromise> DnsDaemon::async_lookup(boost::shared_ptr<SockAddr> sock_addr, std::string host, unsigned port){
	PROFILE_ME;

	AUTO(promise, boost::make_shared<JobPromise>());

	const AUTO(param, new DnsCallbackParam(STD_MOVE(sock_addr), host, port, promise));
	atomic_add(g_pending_callback_count, 1, ATOMIC_RELAXED); // noexcept
	try {
		::sigevent sev;
		sev.sigev_notify            = SIGEV_THREAD;
		sev.sigev_value.sival_ptr   = param;
		sev.sigev_notify_function   = &dns_callback;
		sev.sigev_notify_attributes = NULLPTR;
		AUTO(pcb, &(param->cb));
		const int gai_code = ::getaddrinfo_a(GAI_NOWAIT, &pcb, 1, &sev); // noexcept
		if(gai_code != 0){
			LOG_POSEIDON_WARNING("DNS failure: gai_code = ", gai_code, ", err_msg = ", ::gai_strerror(gai_code));
			DEBUG_THROW(Exception, sslit("DNS failure"));
		}
	} catch(...){
		atomic_sub(g_pending_callback_count, 1, ATOMIC_RELAXED); // noexcept
		delete param;
		throw;
	}

	return promise;
}

}
