// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_
#define POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class JobBase;
class JobPromise;

struct JobDispatcher {
	static void start();
	static void stop();

	// 调用 do_modal() 之后会阻塞直到任意线程调用 quit_modal() 为止。
	static void do_modal();
	static bool is_running();
	static void quit_modal();

	static void enqueue(boost::shared_ptr<JobBase> job, boost::shared_ptr<const bool> withdrawn);
	static void yield(boost::shared_ptr<const JobPromise> promise, bool insignificant);

	static void pump_all();

private:
	JobDispatcher();
};

}

#endif
