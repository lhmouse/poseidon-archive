// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include <signal.h>
#include "log.hpp"
#include "time.hpp"
#include "exception.hpp"
#include "singletons/main_config.hpp"
#include "singletons/timer_daemon.hpp"
#include "singletons/mysql_daemon.hpp"
#include "singletons/epoll_daemon.hpp"
#include "singletons/system_http_server.hpp"
#include "singletons/job_dispatcher.hpp"
#include "singletons/module_depository.hpp"
#include "singletons/event_dispatcher.hpp"
#include "singletons/profile_depository.hpp"
#include "profiler.hpp"

namespace Poseidon {

namespace {
	void sigTermProc(int){
		LOG_POSEIDON_WARNING("Received SIGTERM, will now exit...");
		JobDispatcher::quitModal();
	}

	void sigIntProc(int){
		static const boost::uint64_t KILL_TIMER_EXPIRES = 5000;
		static boost::uint64_t s_killTimer = 0;

		// 系统启动的时候这个时间是从 0 开始的，如果这时候按下 Ctrl+C 就会立即终止。
		// 因此将计时器的起点设为该区间以外。
		const AUTO(now, getFastMonoClock() + KILL_TIMER_EXPIRES + 1);
		if(s_killTimer + KILL_TIMER_EXPIRES < now){
			s_killTimer = now + KILL_TIMER_EXPIRES;
		}
		if(s_killTimer <= now){
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

#define START(x_)	const RaiiSingletonRunner<x_> UNIQUE_ID

	void run(){
		PROFILE_ME;

		START(MySqlDaemon);
		START(JobDispatcher);

		try {
			START(SystemHttpServer);
			START(ModuleDepository);

			START(TimerDaemon);
			START(EpollDaemon);
			START(EventDispatcher);

			const AUTO(initModules, MainConfig::getAll<std::string>("init_module"));
			for(AUTO(it, initModules.begin()); it != initModules.end(); ++it){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Loading init module: ", *it);
				ModuleDepository::load(it->c_str());
			}
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for all asynchronous MySQL operations to complete...");
			MySqlDaemon::waitForAllAsyncOperations();

			JobDispatcher::doModal();
		} catch(...){
			JobDispatcher::pumpAll();
			throw;
		}

		JobDispatcher::pumpAll();
	}
}

}

int main(int argc, char **argv){
	using namespace Poseidon;

	Logger::setThreadTag("P   "); // Primary
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "-------------------------- Starting up -------------------------");

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Setting up signal handlers...");
	::signal(SIGHUP, &sigTermProc);
	::signal(SIGQUIT, &sigTermProc);
	::signal(SIGABRT, &sigTermProc);
	::signal(SIGTERM, &sigTermProc);
	::signal(SIGINT, &sigIntProc);
	::signal(SIGCHLD, SIG_IGN);
	::signal(SIGPIPE, SIG_IGN);

	try {
		MainConfig::setRunPath((1 < argc) ? argv[1] : "/usr/etc/poseidon");
		MainConfig::reload();

		START(ProfileDepository);

		unsigned long long logMask;
		if(MainConfig::get(logMask, "log_mask")){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Setting new log mask: 0x", std::hex, std::uppercase, logMask);
			Logger::setMask(static_cast<boost::uint64_t>(-1), logMask);
		}

		run();

		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "------------- Server has been shut down gracefully -------------");
		return EXIT_SUCCESS;
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("std::exception thrown in main(): what = ", e.what());
	} catch(...){
		LOG_POSEIDON_ERROR("Unknown exception thrown in main().");
	}

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_WARNING, "----------------- Server has exited abnormally -----------------");
	return EXIT_FAILURE;
}
