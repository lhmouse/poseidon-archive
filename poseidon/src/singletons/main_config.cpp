// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "main_config.hpp"
#include "../config_file.hpp"
#include "../log.hpp"
#include "../system_exception.hpp"
#include "../raii.hpp"
#include "../mutex.hpp"
#include <limits.h>
#include <stdlib.h>

namespace Poseidon {

namespace {
	CONSTEXPR const char g_main_conf_name[] = "main.conf";

	struct Real_path_deleter {
		CONSTEXPR char *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(char *ptr) const NOEXCEPT {
			::free(ptr);
		}
	};

	Mutex g_mutex;
	boost::shared_ptr<Config_file> g_config;
}

void Main_config::set_run_path(const char *path){
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Setting new working directory: ", path);
	Unique_handle<Real_path_deleter> real_path;
	DEBUG_THROW_UNLESS(real_path.reset(::realpath(path, NULLPTR)), System_exception);
	LOG_POSEIDON(Logger::special_major | Logger::level_debug, "> Resolved real path: ", real_path);
	DEBUG_THROW_UNLESS(::chdir(real_path.get()) == 0, System_exception);
}
void Main_config::reload(){
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Loading main config file: ", g_main_conf_name);
	AUTO(config, boost::make_shared<Config_file>(g_main_conf_name));
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Done loading main config file: ", g_main_conf_name);
	const Mutex::Unique_lock lock(g_mutex);
	g_config.swap(config);
}

boost::shared_ptr<const Config_file> Main_config::get_file(){
	const Mutex::Unique_lock lock(g_mutex);
	DEBUG_THROW_UNLESS(g_config, Exception, Rcnts::view("Main config file has not been loaded"));
	return g_config;
}

}
