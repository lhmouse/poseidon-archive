// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UTILITIES_HPP_
#define POSEIDON_UTILITIES_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include <vector>
#include <string>
#include <cstddef>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include <errno.h>
#include "shared_nts.hpp"

namespace Poseidon {

// 参考 PHP 的 explode() 函数。
template<typename T>
inline std::vector<T> explode(char separator, const std::string &str, std::size_t limit = 0){
	std::vector<T> ret;
	std::size_t begin = 0;
	std::string temp;
	for(;;){
		const std::size_t end = str.find(separator, begin);
		if(end == std::string::npos){
			if(begin < str.size()){
				temp.assign(str.begin() + begin, str.end());
				ret.push_back(boost::lexical_cast<T>(temp));
			}
			break;
		}
		if(ret.size() == limit - 1){	// 如果 limit 为零则 limit - 1 会变成 SIZE_MAX。
			temp.assign(str.begin() + begin, str.end());
			ret.push_back(boost::lexical_cast<T>(temp));
			break;
		}
		temp.assign(str.begin() + begin, str.begin() + end);
		ret.push_back(boost::lexical_cast<T>(temp));
		begin = end + 1;
	}
	return ret;
}
template<>
inline std::vector<std::string> explode(char separator, const std::string &str, std::size_t limit){
	std::vector<std::string> ret;
	std::size_t begin = 0;
	for(;;){
		const std::size_t end = str.find(separator, begin);
		if(end == std::string::npos){
			if(begin < str.size()){
				ret.push_back(str.substr(begin));
			}
			break;
		}
		if(ret.size() == limit - 1){	// 如果 limit 为零则 limit - 1 会变成 SIZE_MAX。
			ret.push_back(str.substr(begin));
			break;
		}
		ret.push_back(str.substr(begin, end - begin));
		begin = end + 1;
	}
	return ret;
}

template<typename T>
inline std::string implode(char separator, const std::vector<T> &vec){
	std::ostringstream oss;
	for(AUTO(it, vec.begin()); it != vec.end(); ++it){
		oss <<*it;
		if(separator != 0){
			oss.put(separator);
		}
	}
	std::string ret = oss.str();
	if(!ret.empty()){
		ret.erase(ret.end() - 1, ret.end());
	}
	return ret;
}
template<>
inline std::string implode(char separator, const std::vector<std::string> &vec){
	std::string ret;
	for(AUTO(it, vec.begin()); it != vec.end(); ++it){
		ret.append(*it);
		if(separator != 0){
			ret.push_back(separator);
		}
	}
	return ret;
}

boost::uint64_t getUtcTime();		// 单位毫秒。
boost::uint64_t getLocalTime();		// 单位毫秒。
boost::uint64_t getUtcTimeFromLocal(boost::uint64_t local);
boost::uint64_t getLocalTimeFromUtc(boost::uint64_t utc);

boost::uint64_t getMonoClock() NOEXCEPT;	// 单位微秒。

struct DateTime {
	unsigned yr;
	unsigned mon;
	unsigned day;

	unsigned hr;
	unsigned min;
	unsigned sec;

	unsigned ms;
};

DateTime breakDownTime(boost::uint64_t ms);
boost::uint64_t assembleTime(const DateTime &dt);

std::size_t formatTime(char *buffer, std::size_t max, boost::uint64_t ms, bool showMs);
boost::uint64_t scanTime(const char *str);

// 在区间 [lower, upper) 范围内生成伪随机数。
// 前置条件：lower < upper
boost::uint32_t rand32();
boost::uint64_t rand64();
boost::uint32_t rand32(boost::uint32_t lower, boost::uint32_t upper);
double randDouble(double lower = 0.0, double upper = 1.0);

SharedNts getErrorDesc(int errCode = errno) NOEXCEPT;
std::string getErrorDescAsString(int errCode = errno);

}

#endif
