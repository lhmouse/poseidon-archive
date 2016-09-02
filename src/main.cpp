// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include <signal.h>
#include "log.hpp"
#include "time.hpp"
#include "exception.hpp"
#include "singletons/main_config.hpp"
#include "singletons/dns_daemon.hpp"
#include "singletons/timer_daemon.hpp"
#include "singletons/mysql_daemon.hpp"
#include "singletons/mongodb_daemon.hpp"
#include "singletons/epoll_daemon.hpp"
#include "singletons/system_http_server.hpp"
#include "singletons/job_dispatcher.hpp"
#include "singletons/module_depository.hpp"
#include "singletons/event_dispatcher.hpp"
#include "singletons/profile_depository.hpp"
#include "profiler.hpp"

namespace Poseidon {

namespace {
	void sig_term_proc(int){
		LOG_POSEIDON_WARNING("Received SIGTERM, will now exit...");
		JobDispatcher::quit_modal();
	}

	void sig_int_proc(int){
		static const boost::uint64_t KILL_TIMER_EXPIRES = 5000;
		static boost::uint64_t s_kill_timer = 0;

		// 系统启动的时候这个时间是从 0 开始的，如果这时候按下 Ctrl+C 就会立即终止。
		// 因此将计时器的起点设为该区间以外。
		const AUTO(now, get_fast_mono_clock() + KILL_TIMER_EXPIRES + 1);
		if(s_kill_timer + KILL_TIMER_EXPIRES < now){
			s_kill_timer = now + KILL_TIMER_EXPIRES;
		}
		if(s_kill_timer <= now){
			LOG_POSEIDON_FATAL("Received SIGINT, will now terminate abnormally...");
			::raise(SIGKILL);
		} else {
			LOG_POSEIDON_WARNING("Received SIGINT, trying to exit gracefully... If I don't terminate in ",
				KILL_TIMER_EXPIRES, " milliseconds, press ^C again.");
			::raise(SIGTERM);
		}
	}

	template<typename T>
	struct RaiiSingletonRunner : NONCOPYABLE {
		RaiiSingletonRunner(){
			T::start();
		}
		~RaiiSingletonRunner(){
			T::stop();
		}
	};

#define START(x_)   const RaiiSingletonRunner<x_> UNIQUE_ID

	void run(){
		PROFILE_ME;

		const boost::uint64_t CLEANUP_LOG_MASK = Logger::LV_FATAL | Logger::LV_ERROR | Logger::LV_WARNING | Logger::LV_INFO;

		START(DnsDaemon);
		START(MySqlDaemon);
		START(MongoDbDaemon);
		START(JobDispatcher);

		unsigned long long log_mask;
		if(MainConfig::get(log_mask, "log_mask")){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Setting new log mask: 0x", std::hex, std::uppercase, log_mask);
			Logger::set_mask(-1ull, log_mask);
		}
		try {
			START(SystemHttpServer);
			START(ModuleDepository);

			START(TimerDaemon);
			START(EpollDaemon);
			START(EventDispatcher);

			const AUTO(init_modules, MainConfig::get_all<std::string>("init_module"));
			for(AUTO(it, init_modules.begin()); it != init_modules.end(); ++it){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Loading init module: ", *it);
				ModuleDepository::load(it->c_str());
			}
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for all asynchronous MySQL operations to complete...");
			MySqlDaemon::wait_for_all_async_operations();
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for all asynchronous MongoDB operations to complete...");
			MongoDbDaemon::wait_for_all_async_operations();

			JobDispatcher::do_modal();
		} catch(...){
			JobDispatcher::pump_all();
			Logger::set_mask(0, CLEANUP_LOG_MASK);
			throw;
		}
		JobDispatcher::pump_all();
		Logger::set_mask(0, CLEANUP_LOG_MASK);
	}
}

std::terminate_handler g_old_terminate_handler = NULLPTR;

__attribute__((__noreturn__))
void terminate_handler(){
	LOG_POSEIDON_FATAL("std::terminate() is called!!");

	if(std::uncaught_exception()){
		try {
			throw;
		} catch(Exception &e){
			LOG_POSEIDON_FATAL("Poseidon::Exception thrown: what = ", e.what(),
				", file = ", e.get_file(), ", line = ", e.get_line(), ", func = ", e.get_func());
		} catch(std::exception &e){
			LOG_POSEIDON_FATAL("std::exception thrown: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_FATAL("Unknown exception thrown.");
		}
	}

	if(g_old_terminate_handler){
		(*g_old_terminate_handler)();
	}
	std::abort();
}

}

int main(int argc, char **argv){
	using namespace Poseidon;

	g_old_terminate_handler = std::set_terminate(&terminate_handler);

	Logger::set_thread_tag("P   "); // Primary
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "-------------------------- Starting up -------------------------");

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Setting up signal handlers...");
	::signal(SIGHUP, &sig_term_proc);
	::signal(SIGQUIT, &sig_term_proc);
	::signal(SIGTERM, &sig_term_proc);
	::signal(SIGINT, &sig_int_proc);
	::signal(SIGCHLD, SIG_IGN);
	::signal(SIGPIPE, SIG_IGN);

	try {
		MainConfig::set_run_path((1 < argc) ? argv[1] : "/usr/etc/poseidon");
		MainConfig::reload();

		START(ProfileDepository);

		run();
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("std::exception thrown in main(): what = ", e.what());
		goto _failure;
	} catch(...){
		LOG_POSEIDON_ERROR("Unknown exception thrown in main().");
		goto _failure;
	}

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "------------- Server has been shut down gracefully -------------");
	return EXIT_SUCCESS;

_failure:
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_WARNING, "----------------- Server has exited abnormally -----------------");
	return EXIT_FAILURE;
}
