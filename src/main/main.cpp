#include "../precompiled.hpp"
#include "utilities.hpp"
#include "log.hpp"
#include "exception.hpp"
#include "optional_map.hpp"
#include "singletons/session_manager.hpp"
#include "singletons/job_dispatcher.hpp"
#include <fstream>
#include <csignal>
using namespace Poseidon;

namespace {

void sigTermProc(int sig){
    LOG_INFO <<"Received SIGTERM, will now exit...";
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
		LOG_INFO <<"Received SIGINT, trying to exit gracefully...";
		LOG_INFO <<"If I don't terminate in 5 seconds, press ^C again.";
		std::raise(SIGTERM);
	} else {
		LOG_FATAL <<"Received SIGINT, will now terminate abnormally...";
		std::raise(SIGKILL);
	}
}

OptionalMap loadConfig(const char *confPath){
	OptionalMap config;
	std::ifstream ifs(confPath);
	if(!ifs.good()){
		LOG_FATAL <<"Cannot open config file.";
		std::abort();
	}
	std::string line;
	std::size_t count = 0;
	while(std::getline(ifs, line)){
		++count;
		std::size_t pos = line.find('#');
		if(pos != std::string::npos){
			line.resize(pos);
		}
		pos = line.find_first_not_of(" \t");
		if(pos == std::string::npos){
			continue;
		}
		std::size_t equ = line.find('=', pos);
		if(equ == pos){
			LOG_FATAL <<"Error in config file on line " <<count <<": Name expected.";
			std::abort();
		}
		if(equ == std::string::npos){
			LOG_FATAL <<"Error in config file on line " <<count <<": '=' expected.";
			std::abort();
		}

		std::string key = line.substr(pos, equ);
		key.resize(key.find_last_not_of(" \t") + 1);
		pos = line.find_first_not_of(" \t", equ + 1);

		std::string val = line.substr(pos);
		pos = val.find_last_not_of(" \t");
		if(pos == std::string::npos){
			val.clear();
		} else {
			val.resize(pos + 1);
		}

		LOG_DEBUG <<key <<" = " <<val;
		config.set(key, val);
	}
	return config;
}

}

static bool idle1(){
	DEBUG_THROW(Exception, "idle1");
	return true;
}
static bool idle2(){
	static int cnt = 10;
	LOG_WARNING <<"idle2, count = " <<--cnt;
	return cnt;
}

int main(int argc, char **argv){
	LOG_INFO <<"-------------------------- Starting up -------------------------";
SessionManager::registerIdleCallback(idle1);
SessionManager::registerIdleCallback(idle2);
	const char *confPath = "/var/poseidon/config/conf.rc";
	if(1 < argc){
		confPath = argv[1];
	}
	LOG_INFO <<"Loading config from " <<confPath <<"...";
	AUTO(const config, loadConfig(confPath));

	AUTO_REF(logLevel, config["log_level"]);
	if(!logLevel.empty()){
		const int newLevel = boost::lexical_cast<int>(logLevel);
		LOG_INFO <<"Setting log level to " <<newLevel <<", was " <<Log::getLevel() <<"...";
		Log::setLevel(newLevel);
	}

	LOG_INFO <<"Setting up signal handlers...";
	std::signal(SIGINT, sigIntProc);
	std::signal(SIGTERM, sigTermProc);

	LOG_INFO <<"Starting timer daemon...";
	//SessionManager::startDaemon();

	LOG_INFO <<"Starting database daemon...";
	//SessionManager::startDaemon();

	LOG_INFO <<"Starting socket server...";
	//SessionManager::startDaemon();

	LOG_INFO <<"Starting http daemon...";
	//SessionManager::startDaemon();

	LOG_INFO <<"Starting epoll daemon...";
	SessionManager::startDaemon();

	LOG_INFO <<"Entering modal loop...";
	JobDispatcher::doModal();

	LOG_INFO <<"----------------- main() has exited gracefully -----------------";
	return EXIT_SUCCESS;
}

namespace {

// 这部分一定要放在翻译单元末尾。
struct GlobalCleanup : boost::noncopyable {
	~GlobalCleanup() throw() {
		LOG_INFO <<"Stopping epoll daemon...";
		SessionManager::stopDaemon();

		LOG_INFO <<"Stopping database daemon...";
		//SessionManager::stopDaemon();

		LOG_INFO <<"Stopping timer daemon...";
		//SessionManager::stopDaemon();
	}
} const g_globalCleanup;

}
