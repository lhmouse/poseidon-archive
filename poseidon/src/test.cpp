// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "fwd.hpp"
#include "log.hpp"
#include "module_raii.hpp"
#include "async_job.hpp"

using namespace Poseidon;

namespace {
	void async_proc(){
		POSEIDON_LOG_ERROR("hello world!");
	}
}

POSEIDON_MODULE_RAII(handles){
	(void)handles;
	enqueue_async_job(VAL_INIT, &async_proc);
	POSEIDON_LOG_FATAL("hello world!");
}
