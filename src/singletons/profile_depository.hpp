// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PROFILE_DEPOSITORY_HPP_
#define POSEIDON_PROFILE_DEPOSITORY_HPP_

#include "../cxx_ver.hpp"
#include <vector>

namespace Poseidon {

class ProfileDepository {
public:
	struct SnapshotElement {
		const char *file;
		unsigned long line;
		const char *func;
		unsigned long long samples; // 采样数。
		double total; // 控制流进入函数，直到退出函数（正常返回或异常被抛出），经历的总毫秒数。
		double exclusive; // ms_total 扣除执行点位于其他 profiler 之中的毫秒数。
	};

private:
	ProfileDepository();

public:
	static void start();
	static void stop();

	static bool is_enabled() NOEXCEPT;
	static void accumulate(const char *file, unsigned long line, const char *func, bool new_sample, double total, double exclusive) NOEXCEPT;

	static void snapshot(std::vector<SnapshotElement> &ret);
	static void clear() NOEXCEPT;
};

}

#endif
