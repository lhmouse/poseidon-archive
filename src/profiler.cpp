// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "profiler.hpp"
#include "singletons/profile_depository.hpp"
#include "time.hpp"
#include "log.hpp"

namespace Poseidon {

namespace {
	__thread Profiler *t_top_profiler = 0; // XXX: NULLPTR
}

void Profiler::accumulate_all_in_thread() NOEXCEPT {
	const AUTO(top, t_top_profiler);
	if(top){
		const AUTO(now, get_hi_res_mono_clock());
		for(AUTO(cur, top); cur; cur = cur->m_prev){
			cur->accumulate(now);
		}
	}
}

void *Profiler::begin_stack_switch() NOEXCEPT {
	if(!ProfileDepository::is_enabled()){
		return NULLPTR;
	}

	const AUTO(top, t_top_profiler);
	if(top){
		const AUTO(now, get_hi_res_mono_clock());
		top->accumulate(now);
		top->m_yielded_since = now;
	}
	t_top_profiler = NULLPTR;
	return top;
}
void Profiler::end_stack_switch(void *opaque) NOEXCEPT {
	if(!ProfileDepository::is_enabled()){
		return;
	}

	const AUTO(top, static_cast<Profiler *>(opaque));
	if(top){
		const AUTO(now, get_hi_res_mono_clock());
		top->m_excluded += now - top->m_yielded_since;
		top->accumulate(now);
	}
	t_top_profiler = top;
}

Profiler::Profiler(const char *file, unsigned long line, const char *func) NOEXCEPT
	: m_prev(t_top_profiler), m_file(file), m_line(line), m_func(func)
	, m_start(0), m_excluded(0), m_yielded_since(0)
{
	if(!ProfileDepository::is_enabled()){
		return;
	}

	const AUTO(now, get_hi_res_mono_clock());
	m_start = now;

	t_top_profiler = this;
}
Profiler::~Profiler() NOEXCEPT {
	if(!ProfileDepository::is_enabled()){
		return;
	}

	t_top_profiler = m_prev;

	const AUTO(now, get_hi_res_mono_clock());
	accumulate(now);

	if(std::uncaught_exception()){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Exception backtrace: file = ", m_file, ", line = ", m_line, ", func = ", m_func);
	}
}

void Profiler::accumulate(double now) NOEXCEPT {
	const AUTO(total, now - m_start);
	const AUTO(exclusive, total - m_excluded);
	m_start = now;
	m_excluded = 0;

	if(m_prev){
		m_prev->m_excluded += total;
	}

	ProfileDepository::accumulate(m_file, m_line, m_func, total, exclusive);
}

}
