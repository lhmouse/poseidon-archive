#include "../precompiled.hpp"
#include "utilities.hpp"
#include <csignal>
#include "log.hpp"
#include "exception.hpp"
#include "singletons/config_file.hpp"
#include "singletons/database_daemon.hpp"
#include "singletons/timer_daemon.hpp"
#include "singletons/epoll_daemon.hpp"
#include "singletons/job_dispatcher.hpp"
#include "socket_server.hpp"
#include "player_session.hpp"
#include "http_session.hpp"
using namespace Poseidon;

namespace {

void sigTermProc(int sig){
    LOG_WARNING("Received SIGTERM, will now exit...");
    JobDispatcher::quitModal();
}

void sigIntProc(int sig){
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

#define RUN_SING_2_(name_, ln_)		const RaiiSingletonRunner<name_> runner_ ## ln_ ## _
#define RUN_SING_1_(name_, ln_)		RUN_SING_2_(name_, ln_)
#define RUN_SINGLETON(name_)		RUN_SING_1_(name_, __LINE__)

void run(){
	AUTO_REF(logLevel, ConfigFile::get("log_level"));
	if(!logLevel.empty()){
		const int newLevel = boost::lexical_cast<int>(logLevel);
		LOG_WARNING("Setting log level to ", newLevel, ", was ", Log::getLevel(), "...");
		Log::setLevel(newLevel);
	}

	//RUN_SINGLETON(DatabaseDaemon);
	RUN_SINGLETON(TimerDaemon);
	RUN_SINGLETON(EpollDaemon);

	LOG_INFO("Creating player session server...");
	EpollDaemon::addSocketServer(
		boost::make_shared<SocketServer<PlayerSession> >(
			ConfigFile::get("socket_bind"),
			boost::lexical_cast<unsigned>(ConfigFile::get("socket_port"))
		)
	);

	LOG_INFO("Creating http server...");
	EpollDaemon::addSocketServer(
		boost::make_shared<SocketServer<HttpSession> >(
			ConfigFile::get("http_bind"),
			boost::lexical_cast<unsigned>(ConfigFile::get("http_port"))
		)
	);

	LOG_INFO("Entering modal loop...");
	JobDispatcher::doModal();
}

}

int main(int argc, char **argv){
	LOG_INFO("-------------------------- Starting up -------------------------");

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
}
