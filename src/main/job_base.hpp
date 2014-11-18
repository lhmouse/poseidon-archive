// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_JOB_BASE_HPP_
#define POSEIDON_JOB_BASE_HPP_

#include <boost/noncopyable.hpp>
#include <boost/enable_shared_from_this.hpp>

namespace Poseidon {

class JobBase : boost::noncopyable
	, public boost::enable_shared_from_this<JobBase>
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
