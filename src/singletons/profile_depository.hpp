// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PROFILE_DEPOSITORY_HPP_
#define POSEIDON_PROFILE_DEPOSITORY_HPP_

#include "../cxx_ver.hpp"
#include <vector>

namespace Poseidon {

class ProfileDepository {
private:
	ProfileDepository();

public:
	struct SnapshotElement {
		const char *file;
		unsigned long line;
		const char *func;

		// 采样数。
		unsigned long long samples;
		// 控制流进入函数，直到退出函数（正常返回或异常被抛出），经历的总纳秒数。
		unsigned long long ns_total;
		// ns_total 扣除执行点位于其他 profiler 之中的纳秒数。
		unsigned long long ns_exclusive;
	};

	static void start();
	static void stop();

	static bool is_enabled();
	static void accumulate(const char *file, unsigned long line, const char *func, double total, double exclusive) NOEXCEPT;

	static std::vector<SnapshotElement> snapshot();
	static void clear();
};

}

#endif
