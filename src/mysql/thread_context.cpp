// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "thread_context.hpp"
#include "connection.hpp"
#include <mysql.h>
#include "../exception.hpp"
#include "../log.hpp"

namespace Poseidon {

namespace MySql {
	namespace {
		::pthread_once_t g_mysql_once = PTHREAD_ONCE_INIT;

		__thread std::size_t t_init_count = 0;

		void init_mysql(){
			LOG_POSEIDON_INFO("Initializing MySQL library...");

			if(::mysql_library_init(0, NULLPTR, NULLPTR) != 0){
				LOG_POSEIDON_FATAL("Could not initialize MySQL library.");
				std::abort();
			}

			std::atexit(&::mysql_library_end);
		}
	}

	ThreadContext::ThreadContext(){
		if(++t_init_count == 1){
			const int err = ::pthread_once(&g_mysql_once, &init_mysql);
			if(err != 0){
				LOG_POSEIDON_FATAL("::pthread_once() failed with error code ", err);
				std::abort();
			}

			LOG_POSEIDON_INFO("Initializing MySQL thread...");

			if(::mysql_thread_init() != 0){
				LOG_POSEIDON_ERROR("Could not initialize MySQL thread.");
				DEBUG_THROW(Exception, sslit("::mysql_thread_init() failed"));
			}
		}
	}
	ThreadContext::~ThreadContext(){
		if(--t_init_count == 0){
			::mysql_thread_end();
		}
	}
}

}
