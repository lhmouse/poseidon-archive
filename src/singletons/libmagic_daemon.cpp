// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "libmagic_daemon.hpp"
#include "main_config.hpp"
#include "../log.hpp"
#include "../atomic.hpp"
#include "../system_exception.hpp"
#include "../raii.hpp"
#include "../profiler.hpp"
#include <magic.h>

namespace Poseidon {

namespace {
	struct MagicCloser {
		CONSTEXPR ::magic_t operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::magic_t cookie) const NOEXCEPT {
			::magic_close(cookie);
		}
	};

	UniqueHandle<MagicCloser> open_database(const char *file){
		UniqueHandle<MagicCloser> cookie;
		DEBUG_THROW_UNLESS(cookie.reset(::magic_open(MAGIC_MIME_TYPE)), SystemException);
		DEBUG_THROW_UNLESS(::magic_load(cookie.get(), file) == 0, Exception, SharedNts(::magic_error(cookie.get())));
		return cookie;
	}

	const char *checked_look_up(::magic_t cookie, const void *data, std::size_t size){
		const AUTO(desc, ::magic_buffer(cookie, data, size));
		DEBUG_THROW_UNLESS(desc, Exception, SharedNts(::magic_error(cookie)));
		return desc;
	}

	volatile bool g_running = false;
	UniqueHandle<MagicCloser> g_cookie;
}

void LibMagicDaemon::start(){
	if(atomic_exchange(g_running, true, memorder_acq_rel) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting libmagic daemon...");

	const AUTO_REF(database, MainConfig::get<std::string>("libmagic_database", "/usr/share/misc/magic"));
	LOG_POSEIDON_INFO("Loading libmagic database: ", database);
	open_database(database.c_str()).swap(g_cookie);
}
void LibMagicDaemon::stop(){
	if(atomic_exchange(g_running, false, memorder_acq_rel) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping libmagic daemon...");

	g_cookie.reset();
}

const char *LibMagicDaemon::guess_mime_type(const void *data, std::size_t size){
	PROFILE_ME;

	DEBUG_THROW_ASSERT(data);
	DEBUG_THROW_ASSERT(g_cookie);
	return checked_look_up(g_cookie.get(), data, size);
}

}
