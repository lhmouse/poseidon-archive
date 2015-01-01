// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "thread_context.hpp"
#include "exception.hpp"
#include "connection.hpp"
#include <boost/thread/once.hpp>
#include <mysql/mysql.h>
#include <mysql/mysql.h>
#include "../log.hpp"
using namespace Poseidon;

namespace {

boost::once_flag g_mysqlInitFlag;

__thread std::size_t g_initCount = 0;

void initMySql(){
	LOG_POSEIDON_INFO("Initializing MySQL library...");

	if(::mysql_library_init(0, NULLPTR, NULLPTR) != 0){
		LOG_POSEIDON_FATAL("Could not initialize MySQL library.");
		std::abort();
	}

	std::atexit(&::mysql_library_end);
}

}

MySqlThreadContext::MySqlThreadContext(){
	if(++g_initCount == 1){
		boost::call_once(&initMySql, g_mysqlInitFlag);

		LOG_POSEIDON_INFO("Initializing MySQL thread...");

		if(::mysql_thread_init() != 0){
			LOG_POSEIDON_FATAL("Could not initialize MySQL thread.");
			DEBUG_THROW(MySqlException, 99999, SharedNts::observe("::mysql_thread_init() failed"));
		}
	}
}
MySqlThreadContext::~MySqlThreadContext(){
	if(--g_initCount == 0){
		::mysql_thread_end();
	}
}
