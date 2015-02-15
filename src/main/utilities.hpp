// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UTILITIES_HPP_
#define POSEIDON_UTILITIES_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include <vector>
#include <string>
#include <iosfwd>
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
	if(!str.empty()){
		std::size_t begin = 0;
		std::string temp;
		for(;;){
			const std::size_t end = str.find(separator, begin);
			if(end == std::string::npos){
				temp.assign(str.begin() + static_cast<std::ptrdiff_t>(begin), str.end());
				ret.push_back(boost::lexical_cast<T>(temp));
				break;
			}
			if((limit != 0) && (ret.size() == limit - 1)){
				temp.assign(str.begin() + static_cast<std::ptrdiff_t>(begin), str.end());
				ret.push_back(boost::lexical_cast<T>(temp));
				break;
			}
			temp.assign(str.begin() + static_cast<std::ptrdiff_t>(begin), str.begin() + static_cast<std::ptrdiff_t>(end));
			ret.push_back(boost::lexical_cast<T>(temp));
			begin = end + 1;
		}
	}
	return ret;
}
template<>
inline std::vector<std::string> explode(char separator, const std::string &str, std::size_t limit){
	std::vector<std::string> ret;
	if(!str.empty()){
		std::size_t begin = 0;
		std::string temp;
		for(;;){
			const std::size_t end = str.find(separator, begin);
			if(end == std::string::npos){
				temp.assign(str.begin() + static_cast<std::ptrdiff_t>(begin), str.end());
				ret.push_back(STD_MOVE(temp));
				break;
			}
			if((limit != 0) && (ret.size() == limit - 1)){
				temp.assign(str.begin() + static_cast<std::ptrdiff_t>(begin), str.end());
				ret.push_back(STD_MOVE(temp));
				break;
			}
			temp.assign(str.begin() + static_cast<std::ptrdiff_t>(begin), str.begin() + static_cast<std::ptrdiff_t>(end));
			ret.push_back(STD_MOVE(temp));
			begin = end + 1;
		}
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

// 单位毫秒。
boost::uint64_t getUtcTime();
boost::uint64_t getLocalTime();
boost::uint64_t getUtcTimeFromLocal(boost::uint64_t local);
boost::uint64_t getLocalTimeFromUtc(boost::uint64_t utc);

// 单位毫秒。
boost::uint64_t getFastMonoClock() NOEXCEPT;
// 单位秒。
double getHiResMonoClock() NOEXCEPT;

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

struct HexDumper {
	const void *const read;
	const std::size_t size;

	HexDumper(const void *read_, std::size_t size_)
		: read(read_), size(size_)
	{
	}
};

extern std::ostream &operator<<(std::ostream &os, const HexDumper &dumper);

inline std::string toUpperCase(std::string src){
	for(AUTO(it, src.begin()); it != src.end(); ++it){
		if(('a' <= *it) && (*it <= 'z')){
			*it -= 0x20;
		}
	}
	return STD_MOVE(src);
}
inline std::string toLowerCase(std::string src){
	for(AUTO(it, src.begin()); it != src.end(); ++it){
		if(('A' <= *it) && (*it <= 'Z')){
			*it += 0x20;
		}
	}
	return STD_MOVE(src);
}

}

#endif
