// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "dns_daemon.hpp"
#include <netdb.h>
#include <unistd.h>
#include "../log.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../thread.hpp"
#include "../mutex.hpp"
#include "../condition_variable.hpp"
#include "../job_promise.hpp"
#include "../sock_addr.hpp"
#include "../ip_port.hpp"
#include "../raii.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace {
	struct AddrinfoFreeer {
		CONSTEXPR ::addrinfo *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::addrinfo *res) const NOEXCEPT {
			::freeaddrinfo(res);
		}
	};

	SockAddr real_dns_look_up(const std::string &host_raw, unsigned port_raw){
		UniqueHandle<AddrinfoFreeer> res;
		std::string host;
		if(!host_raw.empty() && (host_raw.begin()[0] == '[') && (host_raw.end()[-1] == ']')){
			host.assign(host_raw.begin() + 1, host_raw.end() - 1);
		} else {
			host.assign(host_raw.begin(), host_raw.end());
		}
		char port[16];
		std::sprintf(port, "%u", port_raw);
		::addrinfo *tmp_res;
		const int gai_code = ::getaddrinfo(host.c_str(), port, NULLPTR, &tmp_res);
		if(gai_code != 0){
			const char *const err_msg = ::gai_strerror(gai_code);
			LOG_POSEIDON_DEBUG("DNS lookup failure: host:port = ", host, ":", port, ", gai_code = ", gai_code,
				", err_msg = ", err_msg);
			DEBUG_THROW(Exception, SharedNts(err_msg));
		}
		res.reset(tmp_res);

		SockAddr sock_addr(res.get()->ai_addr, res.get()->ai_addrlen);
		LOG_POSEIDON_DEBUG("DNS lookup success: host:port = ", host, ":", port, ", result = ", IpPort(sock_addr));
		return sock_addr;
	}

	class QueryOperation {
	private:
		const boost::shared_ptr<JobPromiseContainer<SockAddr> > m_promise;

		const std::string m_host;
		const unsigned m_port;

	public:
		QueryOperation(boost::shared_ptr<JobPromiseContainer<SockAddr> > promise,
			std::string host, unsigned port)
			: m_promise(STD_MOVE(promise))
			, m_host(STD_MOVE(host)), m_port(port)
		{ }

	public:
		void execute() const {
			if(m_promise.unique()){
				LOG_POSEIDON_DEBUG("Discarding isolated DNS query: m_host = ", m_host);
				return;
			}

			try {
				m_promise->set_success(real_dns_look_up(m_host, m_port));
			} catch(Exception &e){
				LOG_POSEIDON_INFO("Exception thrown: what = ", e.what());
#ifdef POSEIDON_CXX11
				m_promise->set_exception(std::current_exception());
#else
				m_promise->set_exception(boost::copy_exception(e));
#endif
			} catch(std::exception &e){
				LOG_POSEIDON_INFO("std::exception thrown: what = ", e.what());
#ifdef POSEIDON_CXX11
				m_promise->set_exception(std::current_exception());
#else
				m_promise->set_exception(boost::copy_exception(std::runtime_error(e.what())));
#endif
			}
		}
	};

	volatile bool g_running = false;
	Thread g_thread;

	Mutex g_mutex;
	ConditionVariable g_new_operation;
	boost::container::deque<boost::shared_ptr<QueryOperation> > g_operations;

	bool pump_one_element() NOEXCEPT {
		PROFILE_ME;

		boost::shared_ptr<QueryOperation> operation;
		{
			const Mutex::UniqueLock lock(g_mutex);
			if(!g_operations.empty()){
				operation = g_operations.front();
			}
		}
		if(!operation){
			return false;
		}

		try {
			operation->execute();
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown.");
		}
		const Mutex::UniqueLock lock(g_mutex);
		g_operations.pop_front();
		return true;
	}

	void thread_proc(){
		PROFILE_ME;
		LOG_POSEIDON_INFO("DNS daemon started.");

		unsigned timeout = 0;
		for(;;){
			bool busy;
			do {
				busy = pump_one_element();
				timeout = std::min(timeout * 2u + 1u, !busy * 100u);
			} while(busy);

			Mutex::UniqueLock lock(g_mutex);
			if(!atomic_load(g_running, ATOMIC_CONSUME)){
				break;
			}
			g_new_operation.timed_wait(lock, timeout);
		}

		LOG_POSEIDON_INFO("DNS daemon stopped.");
	}
}

void DnsDaemon::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting DNS daemon...");

	Thread(thread_proc, "   D").swap(g_thread);
}
void DnsDaemon::stop(){
	if(atomic_exchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping DNS daemon...");

	if(g_thread.joinable()){
		g_thread.join();
	}
	g_operations.clear();
}

SockAddr DnsDaemon::look_up(const std::string &host, unsigned port){
	PROFILE_ME;

	return real_dns_look_up(host, port);
}

boost::shared_ptr<const JobPromiseContainer<SockAddr> > DnsDaemon::enqueue_for_looking_up(std::string host, unsigned port){
	PROFILE_ME;

	AUTO(promise, boost::make_shared<JobPromiseContainer<SockAddr> >());
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_operations.push_back(boost::make_shared<QueryOperation>(
			promise, STD_MOVE(host), port));
		g_new_operation.signal();
	}
	return STD_MOVE_IDN(promise);
}

}
