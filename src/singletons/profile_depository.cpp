// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "profile_depository.hpp"
#include <boost/container/flat_map.hpp>
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
	boost::container::flat_map<ProfileKey, ProfileCounters> g_profile;
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

void ProfileDepository::accumulate(const char *file, unsigned long line, const char *func, double total, double exclusive) NOEXCEPT {
	try {
		const Mutex::UniqueLock lock(g_mutex);
		AUTO_REF(counters, g_profile[ProfileKey(file, line, func)]);
		atomic_add(counters.samples,      1,               ATOMIC_RELAXED);
		atomic_add(counters.ns_total,     total * 1e6,     ATOMIC_RELAXED);
		atomic_add(counters.ns_exclusive, exclusive * 1e6, ATOMIC_RELAXED);
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
			const AUTO_REF(key, it->first);
			const AUTO_REF(counters, it->second);
			SnapshotElement elem;
			elem.file         = key.file;
			elem.line         = key.line;
			elem.func         = key.func;
			elem.samples      = atomic_load(counters.samples, ATOMIC_RELAXED);
			elem.ns_total     = atomic_load(counters.ns_total, ATOMIC_RELAXED);
			elem.ns_exclusive = atomic_load(counters.ns_exclusive, ATOMIC_RELAXED);
			ret.push_back(elem);
		}
	}
	return ret;
}
void ProfileDepository::clear(){
	const Mutex::UniqueLock lock(g_mutex);
	g_profile.clear();
}

}
