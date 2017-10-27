// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "profile_depository.hpp"
#include "main_config.hpp"
#include "../mutex.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace {
	struct ProfileKey {
		const char *file;
		unsigned long line;
		const char *func;
	};
	struct ProfileCounters {
		unsigned long long samples;
		double total;
		double exclusive;
	};
	struct ProfileKeyComparator {
		bool operator()(const ProfileKey &lhs, const ProfileKey &rhs) const NOEXCEPT {
			int cmp = std::strcmp(lhs.file, rhs.file);
			if(cmp != 0){
				return cmp < 0;
			}
			return lhs.line < rhs.line;
		}
	};
	typedef boost::container::flat_map<ProfileKey, ProfileCounters, ProfileKeyComparator> ProfileMap;

	bool g_enabled = false;

	Mutex g_mutex;
	ProfileMap g_profile;
}

void ProfileDepository::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting profile depository...");

	g_enabled = MainConfig::get<bool>("profiler_enabled", false);
}
void ProfileDepository::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping profile depository...");

	//
}

bool ProfileDepository::is_enabled(){
	return g_enabled;
}

void ProfileDepository::accumulate(const char *file, unsigned long line, const char *func, bool new_sample, double total, double exclusive) NOEXCEPT
try {
	const Mutex::UniqueLock lock(g_mutex);
	const ProfileKey key = { file, line, func };
	AUTO_REF(counters, g_profile[key]);
	counters.samples += new_sample;
	counters.total += total;
	counters.exclusive += exclusive;
} catch(...){
	//
}

std::vector<ProfileDepository::SnapshotElement> ProfileDepository::snapshot(){
	Profiler::accumulate_all_in_thread();

	std::vector<SnapshotElement> ret;
	{
		const Mutex::UniqueLock lock(g_mutex);
		ret.reserve(g_profile.size());
		for(AUTO(it, g_profile.begin()); it != g_profile.end(); ++it){
			SnapshotElement elem;
			elem.file = it->first.file;
			elem.line = it->first.line;
			elem.func = it->first.func;
			elem.samples = it->second.samples;
			elem.total = it->second.total;
			elem.exclusive = it->second.exclusive;
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
