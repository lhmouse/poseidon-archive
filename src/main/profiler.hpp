#ifndef POSEIDON_PROFILER_HPP_
#define POSEIDON_PROFILER_HPP_

#include "cxx_util.hpp"
#include <boost/noncopyable.hpp>

namespace Poseidon {

class Profiler : boost::noncopyable {
public:
	static void flushProfilersInThread();

private:
	Profiler *const m_prev;
	const char *const m_file;
	const unsigned long m_line;
	const char *const m_func;

	unsigned long long m_start;
	unsigned long long m_exclusiveTotal;
	unsigned long long m_exclusiveStart;

public:
	Profiler(const char *file, unsigned long line, const char *func) NOEXCEPT;
	~Profiler() NOEXCEPT;

private:
	void flush(unsigned long long hint) NOEXCEPT;
};

}

#define PROFILE_ME	\
	const ::Poseidon::Profiler UNIQUE_ID(__FILE__, __LINE__, __PRETTY_FUNCTION__)

#endif
