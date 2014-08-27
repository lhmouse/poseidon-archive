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
};

bool operator<(const ProfileKey &lhs, const ProfileKey &rhs){
	const int fileCmp = std::strcmp(lhs.file, rhs.file);
	if(fileCmp != 0){
		return fileCmp < 0;
	}
	return lhs.line < rhs.line;
}

struct ProfileCounters {
	volatile unsigned long long samples;
	volatile unsigned long long usTotal;
	volatile unsigned long long usExclusive;
};

const ProfileCounters ZERO_ITEM = { 0, 0, 0 };

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

	ProfileKey key;
	key.file = m_file;
	key.line = m_line;
	{
		boost::shared_lock<boost::shared_mutex> slock(g_mutex);
		AUTO(it, g_profile.find(key));
		if(it == g_profile.end()){
			slock.unlock();

			const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
			g_profile.insert(std::make_pair(key, ZERO_ITEM));
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

	ProfileKey key;
	key.file = m_file;
	key.line = m_line;
	{
		const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
		AUTO(it, g_profile.find(key));
		if(it != g_profile.end()){
			atomicAdd(it->second.samples, 1);
			atomicAdd(it->second.usTotal, now - m_start);
			atomicAdd(it->second.usExclusive, m_exclusiveTotal);
		}
	}
}
