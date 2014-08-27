#include "../precompiled.hpp"
#include "profiler.hpp"
#include <map>
#include <cstring>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/locks.hpp>
#include "log.hpp"
#include "atomic.hpp"
#include "utilities.hpp"
using namespace Poseidon;

namespace {

struct ProfileKey {
	const char *file;
	unsigned long line;

	ProfileKey(const char *file_, unsigned long line_)
		: file(file_), line(line_)
	{
	}

	bool operator<(const ProfileKey &rhs) const {
		const int fileCmp = std::strcmp(file, rhs.file);
		if(fileCmp != 0){
			return fileCmp < 0;
		}
		return line < rhs.line;
	}
};

struct ProfileCounters {
	volatile unsigned long long samples;
	volatile unsigned long long usTotal;
	volatile unsigned long long usExclusive;

	ProfileCounters()
		: samples(0), usTotal(0), usExclusive(0)
	{
	}
};

__thread Profiler *g_topProfiler = NULL;

boost::shared_mutex g_mutex;
std::map<ProfileKey, ProfileCounters> g_profile;

}

std::vector<ProfileItem> Profiler::snapshot(){
	std::vector<ProfileItem> ret;
	const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
	ret.reserve(g_profile.size());
	for(AUTO(it, g_profile.begin()); it != g_profile.end(); ++it){
		ProfileItem pi;
		pi.file = it->first.file;
		pi.line = it->first.line;
		pi.samples = atomicLoad(it->second.samples);
		pi.usTotal = atomicLoad(it->second.usTotal);
		pi.usExclusive = atomicLoad(it->second.usExclusive);
		ret.push_back(pi);
	}
	return ret;
}

Profiler::Profiler(const char *file, unsigned long line)
	: m_prev(g_topProfiler), m_file(file), m_line(line)
{
	const AUTO(now, getMonoClock());

	{
		boost::shared_lock<boost::shared_mutex> slock(g_mutex);
		AUTO(it, g_profile.find(ProfileKey(m_file, m_line)));
		if(it != g_profile.end()){
			m_impl = &(it->second);
		} else {
			slock.unlock();
			const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
			m_impl = &(g_profile[ProfileKey(m_file, m_line)]);
		}
	}

	m_start = now;
	m_exclusiveTotal = 0;
	m_exclusiveStart = now;

	if(m_prev){
		m_prev->m_exclusiveTotal += now - m_prev->m_exclusiveStart;
	}
	g_topProfiler = this;
}
Profiler::~Profiler() throw() {
	const AUTO(now, getMonoClock());

	g_topProfiler = m_prev;
	m_exclusiveTotal += now - m_exclusiveStart;
	if(m_prev){
		m_prev->m_exclusiveStart = now;
	}

	const AUTO(cnts, static_cast<ProfileCounters *>(m_impl));
	atomicAdd(cnts->samples, 1);
	atomicAdd(cnts->usTotal, now - m_start);
	atomicAdd(cnts->usExclusive, m_exclusiveTotal);
}
