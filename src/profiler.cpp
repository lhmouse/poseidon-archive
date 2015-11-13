// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "profiler.hpp"
#include "singletons/profile_depository.hpp"
#include "time.hpp"
#include "log.hpp"

namespace Poseidon {

namespace {
	__thread Profiler *t_top_profiler = NULLPTR;
}

void Profiler::flush_profilers_in_thread() NOEXCEPT {
	const AUTO(now, get_hi_res_mono_clock());
	for(AUTO(cur, t_top_profiler); cur; cur = cur->m_prev){
		cur->flush(now);
	}
}

void *Profiler::begin_stack_switch() NOEXCEPT {
	const AUTO(top, t_top_profiler);
	if(top){
		const AUTO(now, get_hi_res_mono_clock());
		top->flush(now);
	}
	t_top_profiler = NULLPTR;
	return top;
}
void Profiler::end_stack_switch(void *opaque) NOEXCEPT {
	t_top_profiler = static_cast<Profiler *>(opaque);
}

Profiler::Profiler(const char *file, unsigned long line, const char *func) NOEXCEPT
	: m_prev(t_top_profiler), m_file(file), m_line(line), m_func(func)
{
	if(ProfileDepository::is_enabled()){
		const AUTO(now, get_hi_res_mono_clock());

		m_start = now;
		m_exclusive_total = 0;
		m_exclusive_start = now;

		if(m_prev){
			m_prev->m_exclusive_total += now - m_prev->m_exclusive_start;
		}
	} else {
		m_start = 0;
	}

	t_top_profiler = this;
}
Profiler::~Profiler() NOEXCEPT {
	t_top_profiler = m_prev;

	if(m_start != 0){
		const AUTO(now, get_hi_res_mono_clock());
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
		m_exclusive_total += now - m_exclusive_start;
		if(m_prev){
			m_prev->m_exclusive_start = now;
		}

		ProfileDepository::accumulate(m_file, m_line, m_func, now - m_start, m_exclusive_total);

		m_start = now;
		m_exclusive_total = 0;
		m_exclusive_start = now;
	}
}

}
