// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "profile_depository.hpp"
#include <map>
#include <cstring>
#include "main_config.hpp"
#include "../mutex.hpp"
#include "../atomic.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace {
	struct ProfileKey {
		const char *file;
		unsigned long line;
		const char *func;

		ProfileKey(const char *file_, unsigned long line_, const char *func_)
			: file(file_), line(line_), func(func_)
		{
		}

		bool operator<(const ProfileKey &rhs) const {
			const int file_cmp = std::strcmp(file, rhs.file);
			if(file_cmp != 0){
				return file_cmp < 0;
			}
			return line < rhs.line;
		}
	};

	struct ProfileCounters {
		volatile unsigned long long samples;
		volatile unsigned long long ns_total;
		volatile unsigned long long ns_exclusive;

		ProfileCounters()
			: samples(0), ns_total(0), ns_exclusive(0)
		{
		}
	};

	bool g_enabled = true;

	Mutex g_mutex;
	std::map<ProfileKey, ProfileCounters> g_profile;
}

void ProfileDepository::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting profile depository...");

	MainConfig::get(g_enabled, "enable_profiler");
	LOG_POSEIDON_DEBUG("Enable profiler = ", g_enabled);
}
void ProfileDepository::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping profile depository...");
}

bool ProfileDepository::is_enabled(){
	return g_enabled;
}

void ProfileDepository::accumulate(const char *file, unsigned long line, const char *func,
	double total, double exclusive) NOEXCEPT
{
	try {
		std::map<ProfileKey, ProfileCounters>::iterator it;
		ProfileKey key(file, line, func);
		{
			const Mutex::UniqueLock lock(g_mutex);
			it = g_profile.find(key);
			if(it != g_profile.end()){
				goto _writeProfile;
			}
		}
		{
			const Mutex::UniqueLock lock(g_mutex);
			it = g_profile.insert(std::make_pair(key, ProfileCounters())).first;
		}

	_writeProfile:
		atomic_add(it->second.samples, 1, ATOMIC_RELAXED);
		atomic_add(it->second.ns_total, total * 1e6, ATOMIC_RELAXED);
		atomic_add(it->second.ns_exclusive, exclusive * 1e6, ATOMIC_RELAXED);

//      LOG_POSEIDON_TRACE("Accumulated profile info: file = ", file, ", line = ", line,
//          ", func = ", func, ", total = ", total, " s, exclusive = ", exclusive, " s");
	} catch(...){
	}
}

std::vector<ProfileDepository::SnapshotElement> ProfileDepository::snapshot(){
	Profiler::accumulate_all_in_thread();

	std::vector<SnapshotElement> ret;
	{
		const Mutex::UniqueLock lock(g_mutex);
		ret.reserve(g_profile.size());
		for(AUTO(it, g_profile.begin()); it != g_profile.end(); ++it){
			SnapshotElement pi;
			pi.file = it->first.file;
			pi.line = it->first.line;
			pi.func = it->first.func;
			pi.samples = atomic_load(it->second.samples, ATOMIC_RELAXED);
			pi.ns_total = atomic_load(it->second.ns_total, ATOMIC_RELAXED);
			pi.ns_exclusive = atomic_load(it->second.ns_exclusive, ATOMIC_RELAXED);
			ret.push_back(pi);
		}
	}
	return ret;
}

}
