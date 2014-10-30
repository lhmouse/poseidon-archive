#include "../precompiled.hpp"
#include "profile_manager.hpp"
#include <map>
#include <cstring>
#include <boost/thread/shared_mutex.hpp>
#include "main_config.hpp"
#include "../atomic.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../shared_ntmbs.hpp"
using namespace Poseidon;

namespace {

struct ProfileKey {
	SharedNtmbs file;
	unsigned long line;
	SharedNtmbs func;

	ProfileKey(SharedNtmbs file_, unsigned long line_, SharedNtmbs func_)
		: file(STD_MOVE(file_)), line(line_), func(STD_MOVE(func_))
	{
	}

	bool operator<(const ProfileKey &rhs) const {
		const int fileCmp = std::strcmp(file.get(), rhs.file.get());
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

bool g_enabled = true;

boost::shared_mutex g_mutex;
std::map<ProfileKey, ProfileCounters> g_profile;

}

void ProfileManager::start(){
	AUTO_REF(conf, MainConfig::getConfigFile());

	conf.get(g_enabled, "enable_profiler");
	LOG_DEBUG("Enable profiler = ", g_enabled);
}
void ProfileManager::stop(){
}

bool ProfileManager::isEnabled(){
	return g_enabled;
}

void ProfileManager::accumulate(const char *file, unsigned long line, const char *func,
	unsigned long long total, unsigned long long exclusive) NOEXCEPT
{
	try {
		std::map<ProfileKey, ProfileCounters>::iterator it;
		ProfileKey key(file, line, func);
		{
			const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
			it = g_profile.find(key);
			if(it != g_profile.end()){
				goto write_profile;
			}
		}
		{
			key.file.forkOwning();
			key.func.forkOwning();

			const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
			it = g_profile.insert(std::make_pair(key, ProfileCounters())).first;
		}

	write_profile:
		atomicAdd(it->second.samples, 1);
		atomicAdd(it->second.usTotal, total);
		atomicAdd(it->second.usExclusive, exclusive);

		LOG_DEBUG("Accumulated profile info: file = ", file, ", line = ", line,
			", func = ", func, ", total = ", total, " us, exclusive = ", exclusive, " us");
	} catch(...){
	}
}

std::vector<ProfileSnapshotItem> ProfileManager::snapshot(){
	Profiler::flushProfilersInThread();

	std::vector<ProfileSnapshotItem> ret;
	{
		const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
		ret.reserve(g_profile.size());
		for(AUTO(it, g_profile.begin()); it != g_profile.end(); ++it){
			ProfileSnapshotItem pi;
			pi.file = it->first.file;
			pi.line = it->first.line;
			pi.func = it->first.func;
			pi.samples = atomicLoad(it->second.samples);
			pi.usTotal = atomicLoad(it->second.usTotal);
			pi.usExclusive = atomicLoad(it->second.usExclusive);
			ret.push_back(pi);
		}
	}
	return ret;
}
