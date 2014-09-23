#include "../precompiled.hpp"
#include "utilities.hpp"
#include <csignal>
#include "log.hpp"
#include "exception.hpp"
#include "singletons/config_file.hpp"
#include "singletons/timer_daemon.hpp"
#include "singletons/mysql_daemon.hpp"
#include "singletons/epoll_daemon.hpp"
#include "singletons/job_dispatcher.hpp"
#include "socket_server.hpp"
#include "player/session.hpp"
#include "http/session.hpp"
using namespace Poseidon;

namespace {

void sigTermProc(int){
	LOG_WARNING("Received SIGTERM, will now exit...");
	JobDispatcher::quitModal();
}

void sigIntProc(int){
	static const unsigned long long SAFETY_TIMER_EXPIRES = 300 * 1000000;
	static unsigned long long s_safetyTimer = 0;

	// 系统启动的时候这个时间是从 0 开始的，如果这时候按下 Ctrl+C 就会立即终止。
	// 因此将计时器的起点设为该区间以外。
	const unsigned long long now = getMonoClock() + SAFETY_TIMER_EXPIRES + 1;
	if(s_safetyTimer + SAFETY_TIMER_EXPIRES < now){
		s_safetyTimer = now + 5 * 1000000;
	}
	if(now < s_safetyTimer){
		LOG_WARNING("Received SIGINT, trying to exit gracefully...");
		LOG_WARNING("If I don't terminate in 5 seconds, press ^C again.");
		std::raise(SIGTERM);
	} else {
		LOG_FATAL("Received SIGINT, will now terminate abnormally...");
		std::raise(SIGKILL);
	}
}

template<typename T>
struct RaiiSingletonRunner : boost::noncopyable {
	RaiiSingletonRunner(){
		T::start();
	}
	~RaiiSingletonRunner(){
		T::stop();
	}
};

#define RUN_SINGLETON(name_)	\
	const RaiiSingletonRunner<name_> UNIQUE_ID

void run(){
	const unsigned logLevel = ConfigFile::get<unsigned>("log_level", Log::LV_INFO);
	LOG_INFO("Setting log level to ", logLevel, "...");
	Log::setLevel(logLevel);

	RUN_SINGLETON(TimerDaemon);
	RUN_SINGLETON(MySqlDaemon);
	RUN_SINGLETON(EpollDaemon);

	LOG_INFO("Creating player session server...");
	EpollDaemon::addSocketServer(
		boost::make_shared<SocketServer<PlayerSession> >(
			ConfigFile::get("socket_bind"), ConfigFile::get<unsigned>("socket_port", 0)
		));

	LOG_INFO("Creating http server...");
	EpollDaemon::addSocketServer(
		boost::make_shared<SocketServer<HttpSession> >(
			ConfigFile::get("http_bind"), ConfigFile::get<unsigned>("http_port", 0)
		));

	LOG_INFO("Entering modal loop...");
	JobDispatcher::doModal();
}

}

int main(int argc, char **argv){
	LOG_INFO("-------------------------- Starting up -------------------------");

	try {
		LOG_INFO("Setting up signal handlers...");
		std::signal(SIGINT, sigIntProc);
		std::signal(SIGTERM, sigTermProc);

		const char *confPath = "/var/poseidon/config/conf.rc";
		if(1 < argc){
			confPath = argv[1];
		}
		ConfigFile::reload(confPath);

		run();

		LOG_INFO("------------- Server has been shut down gracefully -------------");
		return EXIT_SUCCESS;
	} catch(Exception &e){
		LOG_ERROR("Exception thrown in main(): file = ", e.file(),
			", line = ", e.line(), ", what = ", e.what());
	} catch(std::exception &e){
		LOG_ERROR("std::exception thrown in main(): what = ", e.what());
	} catch(...){
		LOG_ERROR("Unknown exception thrown in main().");
	}

	LOG_INFO("----------------- Server has exited abnormally -----------------");
	return EXIT_FAILURE;
}
