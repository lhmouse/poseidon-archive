// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include <signal.h>
#include "log.hpp"
#include "time.hpp"
#include "exception.hpp"
#include "atomic.hpp"
#include "checked_arithmetic.hpp"
#include "singletons/main_config.hpp"
#include "singletons/dns_daemon.hpp"
#include "singletons/timer_daemon.hpp"
#include "singletons/mysql_daemon.hpp"
#include "singletons/mongodb_daemon.hpp"
#include "singletons/epoll_daemon.hpp"
#include "singletons/system_server.hpp"
#include "singletons/job_dispatcher.hpp"
#include "singletons/module_depository.hpp"
#include "singletons/event_dispatcher.hpp"
#include "singletons/filesystem_daemon.hpp"
#include "singletons/profile_depository.hpp"
#include "profiler.hpp"

namespace Poseidon {

namespace {
	volatile bool g_running = true;

	void sigterm_proc(int){
		LOG_POSEIDON_WARNING("Received SIGTERM, will now exit...");
		atomic_store(g_running, false, ATOMIC_RELEASE);
	}

	void sighup_proc(int){
		LOG_POSEIDON_WARNING("Received SIGHUP, will now exit...");
		atomic_store(g_running, false, ATOMIC_RELEASE);
	}

	void sigint_proc(int){
		static const boost::uint64_t KILL_TIMEOUT = 5000;
		static const boost::uint64_t RESET_TIMEOUT = 10000;
		static boost::uint64_t s_kill_timer = 0;

		// 系统启动的时候这个时间是从 0 开始的，如果这时候按下 Ctrl+C 就会立即终止。
		// 因此将计时器的起点设为该区间以外。
		const AUTO(virtual_now, saturated_add(get_fast_mono_clock(), RESET_TIMEOUT));
		if(saturated_sub(virtual_now, s_kill_timer) >= RESET_TIMEOUT){
			s_kill_timer = virtual_now;
		}
		if(saturated_sub(virtual_now, s_kill_timer) >= KILL_TIMEOUT){
			LOG_POSEIDON_FATAL("--------------------- Process was killed ----------------------");
			std::_Exit(EXIT_FAILURE);
		}
		LOG_POSEIDON_WARNING("Received SIGINT, trying to exit gracefully...");
		LOG_POSEIDON_WARNING("  If I don't terminate in ", KILL_TIMEOUT, " milliseconds, press ^C again.");
		atomic_store(g_running, false, ATOMIC_RELEASE);
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

		START(DnsDaemon);
		START(FileSystemDaemon);
		START(MySqlDaemon);
		START(MongoDbDaemon);

#ifdef POSEIDON_CXX11
		std::exception_ptr ep;
#endif
		try {
			START(JobDispatcher);

#ifdef POSEIDON_CXX11
			try
#endif
			{
				START(ModuleDepository);
				START(TimerDaemon);
				START(EpollDaemon);
				START(EventDispatcher);
				START(SystemServer);

				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for daemon initialization to complete...");
				::timespec req;
				req.tv_sec = 0;
				req.tv_nsec = 200000000;
				::nanosleep(&req, NULLPTR);

				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Setting new log mask...");
				Logger::initialize_mask_from_config();

				const AUTO(init_modules, MainConfig::get_all<std::string>("init_module"));
				for(AUTO(it, init_modules.begin()); it != init_modules.end(); ++it){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Loading init module: ", *it);
					ModuleDepository::load(it->c_str());
				}

				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for all asynchronous MySQL operations to complete...");
				MySqlDaemon::wait_for_all_async_operations();
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for all asynchronous MongoDB operations to complete...");
				MongoDbDaemon::wait_for_all_async_operations();

				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Entering modal loop...");
				JobDispatcher::do_modal(g_running);
			}
#ifdef POSEIDON_CXX11
			catch(...){
				ep = std::current_exception();
			}
#endif
			Logger::finalize_mask();
		} catch(...){
			Logger::finalize_mask();
			throw;
		}
#ifdef POSEIDON_CXX11
		if(ep){
			std::rethrow_exception(ep);
		}
#endif
	}
}

}

using namespace Poseidon;

int main(int argc, char **argv)
try {
	Logger::set_thread_tag("P   "); // Primary

	::signal(SIGHUP, &sighup_proc);
	::signal(SIGTERM, &sigterm_proc);
	::signal(SIGINT, &sigint_proc);
	::signal(SIGCHLD, SIG_IGN);
	::signal(SIGPIPE, SIG_IGN);

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "------------------------- Starting up -------------------------");

	const char *run_path;
	if(argc > 1){
		run_path = argv[1];
	} else{
		run_path = "/usr/etc/poseidon";
	}
	MainConfig::set_run_path(run_path);
	MainConfig::reload();

	START(ProfileDepository);
	run();

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "------------------ Process exited gracefully ------------------");
	return EXIT_SUCCESS;
} catch(std::exception &e){
	LOG_POSEIDON_ERROR("std::exception thrown in main(): what = ", e.what());

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_ERROR, "------------------ Process exited abnormally ------------------");
	return EXIT_FAILURE;
} catch(...){
	LOG_POSEIDON_ERROR("Unknown exception thrown in main().");

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_ERROR, "------------------ Process exited abnormally ------------------");
	return EXIT_FAILURE;
}
