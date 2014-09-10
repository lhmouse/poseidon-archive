#ifndef POSEIDON_PROFILER_HPP_
#define POSEIDON_PROFILER_HPP_

#include "../cxx_util.hpp"
#include <vector>
#include <boost/noncopyable.hpp>

namespace Poseidon {

struct ProfileItem {
	const char *file;
	unsigned long line;

	// 采样数。
	unsigned long long samples;
	// 控制流进入函数，直到退出函数（正常返回或异常被抛出），经历的总微秒数。
	unsigned long long usTotal;
	// usTotal 扣除执行点位于其他 profiler 之中的微秒数。
	unsigned long long usExclusive;
};

class Profiler : boost::noncopyable {
public:
	static std::vector<ProfileItem> snapshot();

private:
	Profiler *const m_prev;
	const char *const m_file;
	const unsigned long m_line;

	void *m_impl;

	unsigned long long m_start;
	unsigned long long m_exclusiveTotal;
	unsigned long long m_exclusiveStart;

public:
	Profiler(const char *file, unsigned long line);
	~Profiler() NOEXCEPT;
};

}

#define PROFILE_ME	\
	const ::Poseidon::Profiler UNIQUE_ID(__FILE__, __LINE__)

#endif
