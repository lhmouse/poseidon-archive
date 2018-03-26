// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "magic_daemon.hpp"
#include "main_config.hpp"
#include "../log.hpp"
#include "../atomic.hpp"
#include "../system_exception.hpp"
#include "../raii.hpp"
#include "../profiler.hpp"
#include <magic.h>

namespace Poseidon {

namespace {
	struct Magic_closer {
		CONSTEXPR ::magic_t operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::magic_t cookie) const NOEXCEPT {
			::magic_close(cookie);
		}
	};

	Unique_handle<Magic_closer> open_database(const char *file){
		Unique_handle<Magic_closer> cookie;
		DEBUG_THROW_UNLESS(cookie.reset(::magic_open(MAGIC_MIME_TYPE)), System_exception);
		DEBUG_THROW_UNLESS(::magic_load(cookie.get(), file) == 0, Exception, Shared_nts(::magic_error(cookie.get())));
		return cookie;
	}

	const char *checked_look_up(::magic_t cookie, const void *data, std::size_t size){
		const AUTO(desc, ::magic_buffer(cookie, data, size));
		DEBUG_THROW_UNLESS(desc, Exception, Shared_nts(::magic_error(cookie)));
		return desc;
	}

	volatile bool g_running = false;
	Unique_handle<Magic_closer> g_cookie;
}

void Magic_daemon::start(){
	if(atomic_exchange(g_running, true, memory_order_acq_rel) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Starting magic daemon...");

	const AUTO_REF(database, Main_config::get<std::string>("magic_database", "/usr/share/misc/magic"));
	LOG_POSEIDON_INFO("Loading magic database: ", database);
	open_database(database.c_str()).swap(g_cookie);
}
void Magic_daemon::stop(){
	if(atomic_exchange(g_running, false, memory_order_acq_rel) == false){
		return;
	}
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Stopping magic daemon...");

	g_cookie.reset();
}

const char *Magic_daemon::guess_mime_type(const void *data, std::size_t size){
	PROFILE_ME;

	DEBUG_THROW_ASSERT(data);
	DEBUG_THROW_ASSERT(g_cookie);
	return checked_look_up(g_cookie.get(), data, size);
}

}
