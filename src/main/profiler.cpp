// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "profiler.hpp"
#include "singletons/profile_depository.hpp"
#include "utilities.hpp"
using namespace Poseidon;

namespace {

__thread Profiler *t_topProfiler = 0;

}

void Profiler::flushProfilersInThread(){
	const AUTO(now, getMonoClock());
	for(AUTO(cur, t_topProfiler); cur; cur = cur->m_prev){
		cur->flush(now);
	}
}

Profiler::Profiler(const char *file, unsigned long line, const char *func) NOEXCEPT
	: m_prev(t_topProfiler), m_file(file), m_line(line), m_func(func)
{
	if(ProfileDepository::isEnabled()){
		const AUTO(now, getMonoClock());

		m_start = now;
		m_exclusiveTotal = 0;
		m_exclusiveStart = now;

		if(m_prev){
			m_prev->m_exclusiveTotal += now - m_prev->m_exclusiveStart;
		}
	} else {
		m_start = 0;
	}

	t_topProfiler = this;
}
Profiler::~Profiler() NOEXCEPT {
	t_topProfiler = m_prev;

	if(m_start != 0){
		const AUTO(now, getMonoClock());
		flush(now);
		if(m_prev){
			m_prev->flush(now);
		}
	}
}

void Profiler::flush(unsigned long long hint) NOEXCEPT {
	if(m_start != 0){
		m_exclusiveTotal += hint - m_exclusiveStart;
		if(m_prev){
			m_prev->m_exclusiveStart = hint;
		}

		ProfileDepository::accumulate(m_file, m_line, m_func, hint - m_start, m_exclusiveTotal);

		m_start = hint;
		m_exclusiveTotal = 0;
		m_exclusiveStart = hint;
	}
}
