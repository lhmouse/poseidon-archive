#ifndef POSEIDON_PROFILE_MANAGER_HPP_
#define POSEIDON_PROFILE_MANAGER_HPP_

#include "../../cxx_ver.hpp"
#include "../../cxx_util.hpp"
#include <vector>

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

struct ProfileManager {
	static void start();
	static void stop();

	static bool isEnabled();
	static void accumulate(const char *file, unsigned long line,
		unsigned long long total, unsigned long long exclusive) NOEXCEPT;

	static std::vector<ProfileItem> snapshot();

private:
	ProfileManager();
};

}

#endif
