#ifndef POSEIDON_JOB_BASE_HPP_
#define POSEIDON_JOB_BASE_HPP_

#include <boost/enable_shared_from_this.hpp>

namespace Poseidon {

class JobBase
	: public boost::enable_shared_from_this<JobBase>
{
public:
	virtual ~JobBase();

public:
	virtual void perform() = 0;

public:
	// 加到全局队列中（外部不可见），线程安全的。
	void pend();
};

}

#endif
