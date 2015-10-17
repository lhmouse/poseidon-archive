// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_
#define POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace Poseidon {

class JobBase;
class JobPromise;

struct JobDispatcher {
	static void start();
	static void stop();

	// 调用 doModal() 之后会阻塞直到任意线程调用 quitModal() 为止。
	static void doModal();
	static bool isRunning();
	static void quitModal();

	static void enqueue(boost::shared_ptr<JobBase> job,
		boost::shared_ptr<const JobPromise> promise, boost::shared_ptr<const bool> withdrawn);
	static void yield(boost::shared_ptr<const JobPromise> promise);
	static void detachYieldable() NOEXCEPT;

	static void pumpAll();

private:
	JobDispatcher();
};

}

#endif
