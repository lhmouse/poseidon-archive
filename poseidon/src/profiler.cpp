// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "profiler.hpp"
#include "singletons/profile_depository.hpp"
#include "time.hpp"
#include "log.hpp"

namespace Poseidon {

namespace {
	__thread Profiler *t_top = 0; // XXX: NULLPTR
}

void Profiler::accumulate_all_in_thread() NOEXCEPT {
	if(!Profile_depository::is_enabled()){
		return;
	}
	Profiler *cur = t_top;
	if(!cur){
		return;
	}
	const AUTO(now, get_hi_res_mono_clock());
	do {
		cur->accumulate(now, false);
		cur = cur->m_prev;
	} while(cur);
}

void *Profiler::begin_stack_switch() NOEXCEPT {
	if(!Profile_depository::is_enabled()){
		return NULLPTR;
	}
	Profiler *cur = t_top;
	if(!cur){
		return NULLPTR;
	}
	const AUTO(now, get_hi_res_mono_clock());
	cur->accumulate(now, false);
	cur->m_yielded_since = now;
	t_top = NULLPTR;
	return cur;
}
void Profiler::end_stack_switch(void *opaque) NOEXCEPT {
	Profiler *cur = static_cast<Profiler *>(opaque);
	if(!cur){
		return;
	}
	const AUTO(now, get_hi_res_mono_clock());
	cur->m_excluded += now - cur->m_yielded_since;
	cur->accumulate(now, false);
	t_top = cur;
}

Profiler::Profiler(const char *file, unsigned long line, const char *func) NOEXCEPT
	: m_prev(t_top), m_file(file), m_line(line), m_func(func)
	, m_start(0), m_excluded(0), m_yielded_since(0)
{
	if(Profile_depository::is_enabled()){
		const AUTO(now, get_hi_res_mono_clock());
		m_start = now;
		t_top = this;
	}
}
Profiler::~Profiler() NOEXCEPT {
	if(std::uncaught_exception()){
		POSEIDON_LOG(Logger::special_major | Logger::level_info, "Exception backtrace: file = ", m_file, ", line = ", m_line, ", func = ", m_func);
	}

	if(t_top == this){
		const AUTO(now, get_hi_res_mono_clock());
		t_top = m_prev;
		accumulate(now, true);
	}
}

void Profiler::accumulate(double now, bool new_sample) NOEXCEPT {
	const AUTO(total, now - m_start);
	const AUTO(exclusive, total - m_excluded);
	m_start = now;
	m_excluded = 0;

	if(m_prev){
		m_prev->m_excluded += total;
	}

	Profile_depository::accumulate(m_file, m_line, m_func, new_sample, total, exclusive);
}

}
