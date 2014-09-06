#ifndef POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_
#define POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_

#include <boost/shared_ptr.hpp>
#include "../virtual_shared_from_this.hpp"

namespace Poseidon {

class JobBase
	: public virtual VirtualSharedFromThis
{
public:
	virtual ~JobBase();

public:
	virtual void perform() const = 0;

public:
	// 加到全局队列中（外部不可见），线程安全的。
	void pend() const;
};

struct JobDispatcher {
	// 调用 doModal() 之后会阻塞直到任意线程调用 quitModal() 为止。
	static void doModal();
	static void quitModal();

private:
	JobDispatcher();
};

}

#endif
