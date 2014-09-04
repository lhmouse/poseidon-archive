#include "../precompiled.hpp"
#include "utilities.hpp"
#include <csignal>
#include "log.hpp"
#include "exception.hpp"
#include "singletons/config_file.hpp"
#include "singletons/timer_daemon.hpp"
#include "singletons/database_daemon.hpp"
#include "singletons/epoll_daemon.hpp"
#include "singletons/job_dispatcher.hpp"
#include "socket_server.hpp"
#include "player_session.hpp"
#include "http_session.hpp"
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
	AUTO_REF(logLevel, ConfigFile::get("log_level"));
	if(!logLevel.empty()){
		const int newLevel = boost::lexical_cast<int>(logLevel);
		LOG_WARNING("Setting log level to ", newLevel, ", was ", Log::getLevel(), "...");
		Log::setLevel(newLevel);
	}

	RUN_SINGLETON(TimerDaemon);
	//RUN_SINGLETON(DatabaseDaemon);
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

#define PROTOCOL_NAMESPACE  TestNs
#define PROTOCOL_NAME       TestProtocol
#define PROTOCOL_FIELDS     \
	FIELD_VINT50(num)   \
	FIELD_ARRAY(array1,	\
		FIELD_VUINT50(num1)	\
		FIELD_VINT50(real)	\
	)	\
	FIELD_STRING(str)

#include "protocol/protocol_generator.hpp"
#include "singletons/player_servlet_manager.hpp"

static void servletProc(const boost::shared_ptr<PlayerSession> &, StreamBuffer){
	TestNs::TestProtocol p1, p2;
	StreamBuffer buf;
	p1.num = 123;
	p1.array1.resize(2);
	p1.array1[0].num1 = 45;
	p1.array1[0].real = 678;
	p1.array1[1].num1 = 901;
	p1.array1[1].real = 2345;
	p1.str = "meow";
	p1 >> buf;
	p2 << buf;
	LOG_DEBUG("num = ", p2.num,
		", array[0].num1 = ", p2.array1.at(0).num1,
		", array[0].real = ", p2.array1.at(0).real,
		", array[1].num1 = ", p2.array1.at(1).num1,
		", array[1].real = ", p2.array1.at(1).real,
		", str = ", p2.str
	);
}
const AUTO(srv, PlayerServletManager::registerServlet(100, boost::weak_ptr<void>(), servletProc));

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
