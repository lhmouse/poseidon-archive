// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_JOB_BASE_HPP_
#define POSEIDON_JOB_BASE_HPP_

#include "cxx_util.hpp"
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class JobBase : NONCOPYABLE {
public:
	virtual ~JobBase();

public:
	virtual void perform() = 0;
};

// 加到全局队列中（外部不可见），线程安全的。
extern void pendJob(boost::shared_ptr<JobBase> job);

}

#endif
