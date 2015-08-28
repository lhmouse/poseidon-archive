// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "profiler.hpp"
#include "singletons/profile_depository.hpp"
#include "time.hpp"
#include "log.hpp"

namespace Poseidon {

namespace {
	__thread Profiler *t_topProfiler = NULLPTR;
}

void Profiler::flushProfilersInThread() NOEXCEPT {
	const AUTO(now, getHiResMonoClock());
	for(AUTO(cur, t_topProfiler); cur; cur = cur->m_prev){
		cur->flush(now);
	}
}

void *Profiler::beginStackSwitch() NOEXCEPT {
	const AUTO(top, t_topProfiler);
	if(top){
		const AUTO(now, getHiResMonoClock());
		top->flush(now);
	}
	t_topProfiler = NULLPTR;
	return top;
}
void Profiler::endStackSwitch(void *opaque) NOEXCEPT {
	t_topProfiler = static_cast<Profiler *>(opaque);
}

Profiler::Profiler(const char *file, unsigned long line, const char *func) NOEXCEPT
	: m_prev(t_topProfiler), m_file(file), m_line(line), m_func(func)
{
	if(ProfileDepository::isEnabled()){
		const AUTO(now, getHiResMonoClock());

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
		const AUTO(now, getHiResMonoClock());
		flush(now);
		if(m_prev){
			m_prev->flush(now);
		}
	}

	if(std::uncaught_exception()){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Exception backtrace: file = ", m_file, ", line = ", m_line, ", func = ", m_func);
	}
}

void Profiler::flush(double now) NOEXCEPT {
	if(m_start != 0){
		m_exclusiveTotal += now - m_exclusiveStart;
		if(m_prev){
			m_prev->m_exclusiveStart = now;
		}

		ProfileDepository::accumulate(m_file, m_line, m_func, now - m_start, m_exclusiveTotal);

		m_start = now;
		m_exclusiveTotal = 0;
		m_exclusiveStart = now;
	}
}

}
