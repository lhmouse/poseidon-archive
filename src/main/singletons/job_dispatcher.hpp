#ifndef POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_
#define POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_

#include <boost/shared_ptr.hpp>

namespace Poseidon {

class JobBase;

struct JobDispatcher {
	static void start();
	static void stop();

	// 调用 doModal() 之后会阻塞直到任意线程调用 quitModal() 为止。
	static void doModal();
	static void quitModal();

	static void pend(boost::shared_ptr<JobBase> job);

private:
	JobDispatcher();
};

}

#endif
