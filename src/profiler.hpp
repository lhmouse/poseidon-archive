// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PROFILER_HPP_
#define POSEIDON_PROFILER_HPP_

#include "cxx_util.hpp"

namespace Poseidon {

class Profiler : NONCOPYABLE {
public:
	static void accumulate_all_in_thread() NOEXCEPT;

	static void *begin_stack_switch() NOEXCEPT;
	static void end_stack_switch(void *opaque) NOEXCEPT;

private:
	Profiler *const m_prev;
	const char *const m_file;
	const unsigned long m_line;
	const char *const m_func;

	double m_start;
	double m_excluded;
	double m_yielded_since;

public:
	Profiler(const char *file, unsigned long line, const char *func) NOEXCEPT;
	~Profiler() NOEXCEPT;

private:
	void accumulate(double now) NOEXCEPT;
};

}

#define PROFILE_ME  \
	const ::Poseidon::Profiler UNIQUE_ID(__FILE__, __LINE__, __PRETTY_FUNCTION__)

#endif
