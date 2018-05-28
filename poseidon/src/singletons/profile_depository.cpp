// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "profile_depository.hpp"
#include "main_config.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace {
	struct Profile_key {
		const char *file;
		unsigned long line;
		const char *func;
	};
	struct Profile_counters {
		unsigned long long samples;
		double total;
		double exclusive;
	};
	struct Profile_key_comparator {
		bool operator()(const Profile_key &lhs, const Profile_key &rhs) const NOEXCEPT {
			int cmp = std::strcmp(lhs.file, rhs.file);
			if(cmp != 0){
				return cmp < 0;
			}
			return lhs.line < rhs.line;
		}
	};
	typedef boost::container::flat_map<Profile_key, Profile_counters, Profile_key_comparator> Profile_map;

	bool g_enabled = false;

	std::mutex g_mutex;
	Profile_map g_profile;
}

void Profile_depository::start(){
	POSEIDON_LOG(Logger::special_major | Logger::level_info, "Starting profile depository...");

	g_enabled = Main_config::get<bool>("profiler_enabled", false);
}
void Profile_depository::stop(){
	POSEIDON_LOG(Logger::special_major | Logger::level_info, "Stopping profile depository...");

	const std::lock_guard<std::mutex> lock(g_mutex);
	g_profile.clear();
}

bool Profile_depository::is_enabled() NOEXCEPT {
	return g_enabled;
}

void Profile_depository::accumulate(const char *file, unsigned long line, const char *func, bool new_sample, double total, double exclusive) NOEXCEPT
try {
	const std::lock_guard<std::mutex> lock(g_mutex);
	const Profile_key key = { file, line, func };
	AUTO_REF(counters, g_profile[key]);
	counters.samples += new_sample;
	counters.total += total;
	counters.exclusive += exclusive;
} catch(...){
	//
}

void Profile_depository::snapshot(boost::container::vector<Profile_depository::Snapshot_element> &ret){
	Profiler::accumulate_all_in_thread();

	const std::lock_guard<std::mutex> lock(g_mutex);
	ret.reserve(ret.size() + g_profile.size());
	for(AUTO(it, g_profile.begin()); it != g_profile.end(); ++it){
		Snapshot_element elem = { };
		elem.file = it->first.file;
		elem.line = it->first.line;
		elem.func = it->first.func;
		elem.samples = it->second.samples;
		elem.total = it->second.total;
		elem.exclusive = it->second.exclusive;
		ret.push_back(STD_MOVE(elem));
	}
}
void Profile_depository::clear() NOEXCEPT {
	const std::lock_guard<std::mutex> lock(g_mutex);
	g_profile.clear();
}

}
