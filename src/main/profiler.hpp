#ifndef POSEIDON_PROFILER_HPP_
#define POSEIDON_PROFILER_HPP_

#include "../cxx_util.hpp"
#include <boost/noncopyable.hpp>

namespace Poseidon {

class Profiler : boost::noncopyable {
private:
	Profiler *const m_prev;
	const char *const m_file;
	const unsigned long m_line;

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
