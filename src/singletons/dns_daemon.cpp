// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "dns_daemon.hpp"
#include "../log.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../thread.hpp"
#include "../mutex.hpp"
#include "../condition_variable.hpp"
#include "../ip_port.hpp"
#include "../raii.hpp"
#include "../profiler.hpp"
#include <netdb.h>
#include <unistd.h>

namespace Poseidon {

template class PromiseContainer<SockAddr>;

namespace {
	struct AddrinfoFreeer {
		CONSTEXPR ::addrinfo *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::addrinfo *res) const NOEXCEPT {
			::freeaddrinfo(res);
		}
	};

	SockAddr real_dns_look_up(const std::string &host_raw, boost::uint16_t port_raw, bool prefer_ipv4){
		UniqueHandle<AddrinfoFreeer> res;
		std::string host;
		if(!host_raw.empty() && (host_raw.begin()[0] == '[') && (host_raw.end()[-1] == ']')){
			host.assign(host_raw.begin() + 1, host_raw.end() - 1);
		} else {
			host.assign(host_raw.begin(), host_raw.end());
		}
		char port[16];
		std::sprintf(port, "%u", port_raw);
		::addrinfo *res_ptr;
		const int gai_code = ::getaddrinfo(host.c_str(), port, NULLPTR, &res_ptr);
		if(gai_code != 0){
			const char *const err_msg = ::gai_strerror(gai_code);
			LOG_POSEIDON_DEBUG("DNS lookup failure: host:port = ", host, ":", port, ", gai_code = ", gai_code, ", err_msg = ", err_msg);
			DEBUG_THROW(Exception, SharedNts(err_msg));
		}
		DEBUG_THROW_ASSERT(res.reset(res_ptr));

		::addrinfo *res_ptr_ipv4 = NULLPTR;
		::addrinfo *res_ptr_ipv6 = NULLPTR;
		while(res_ptr){
			switch(res_ptr->ai_family){
			case AF_INET:
				res_ptr_ipv4 = res_ptr;
				break;
			case AF_INET6:
				res_ptr_ipv6 = res_ptr;
				break;
			}
			res_ptr = res_ptr->ai_next;
		}
		if(prefer_ipv4){
			res_ptr = res_ptr_ipv4;
		} else {
			res_ptr = res_ptr_ipv6;
		}
		if(!res_ptr){
			res_ptr = res.get();
		}
		SockAddr sock_addr(res_ptr->ai_addr, res_ptr->ai_addrlen);
		LOG_POSEIDON_DEBUG("DNS lookup success: host:port = ", host, ":", port, ", result = ", IpPort(sock_addr));
		return sock_addr;
	}

	volatile bool g_running = false;
	Thread g_thread;

	struct RequestElement {
		boost::weak_ptr<PromiseContainer<SockAddr> > weak_promise;
		std::string host;
		boost::uint16_t port;
		bool prefer_ipv4;
	};

	Mutex g_mutex;
	ConditionVariable g_new_request;
	boost::container::deque<RequestElement> g_queue;

	bool pump_one_element() NOEXCEPT {
		PROFILE_ME;

		RequestElement *elem;
		{
			const Mutex::UniqueLock lock(g_mutex);
			if(g_queue.empty()){
				return false;
			}
			if(g_queue.front().weak_promise.expired()){
				g_queue.pop_front();
				return true;
			}
			elem = &g_queue.front();
		}
		SockAddr sock_addr;
		STD_EXCEPTION_PTR except;
		try {
			sock_addr = real_dns_look_up(elem->host, elem->port, elem->prefer_ipv4);
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
			except = STD_CURRENT_EXCEPTION();
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown.");
			except = STD_CURRENT_EXCEPTION();
		}
		const AUTO(promise, elem->weak_promise.lock());
		if(promise){
			if(except){
				promise->set_exception(STD_MOVE(except), false);
			} else {
				promise->set_success(STD_MOVE(sock_addr), false);
			}
		}
		const Mutex::UniqueLock lock(g_mutex);
		g_queue.pop_front();
		return true;
	}

	void thread_proc(){
		PROFILE_ME;
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "DNS daemon started.");

		unsigned timeout = 0;
		for(;;){
			bool busy;
			do {
				busy = pump_one_element();
				timeout = std::min(timeout * 2u + 1u, !busy * 100u);
			} while(busy);

			Mutex::UniqueLock lock(g_mutex);
			if(!atomic_load(g_running, memorder_consume)){
				break;
			}
			g_new_request.timed_wait(lock, timeout);
		}

		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "DNS daemon stopped.");
	}
}

void DnsDaemon::start(){
	if(atomic_exchange(g_running, true, memorder_acq_rel) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting DNS daemon...");

	Thread(&thread_proc, sslit("   D"), sslit("DNS")).swap(g_thread);
}
void DnsDaemon::stop(){
	if(atomic_exchange(g_running, false, memorder_acq_rel) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping DNS daemon...");

	if(g_thread.joinable()){
		g_thread.join();
	}

	const Mutex::UniqueLock lock(g_mutex);
	g_queue.clear();
}

SockAddr DnsDaemon::look_up(const std::string &host, boost::uint16_t port, bool prefer_ipv4){
	PROFILE_ME;

	return real_dns_look_up(host, port, prefer_ipv4);
}

boost::shared_ptr<const PromiseContainer<SockAddr> > DnsDaemon::enqueue_for_looking_up(std::string host, boost::uint16_t port, bool prefer_ipv4){
	PROFILE_ME;

	AUTO(promise, boost::make_shared<PromiseContainer<SockAddr> >());
	{
		const Mutex::UniqueLock lock(g_mutex);
		RequestElement elem = { promise, STD_MOVE(host), port, prefer_ipv4 };
		g_queue.push_back(STD_MOVE(elem));
		g_new_request.signal();
	}
	return STD_MOVE_IDN(promise);
}

}
