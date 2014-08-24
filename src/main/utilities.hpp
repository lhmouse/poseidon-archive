#ifndef POSEIDON_UTILITIES_HPP_
#define POSEIDON_UTILITIES_HPP_

#include <vector>
#include <string>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include <errno.h>

namespace Poseidon {

template<typename Type>
inline std::vector<Type> split(const std::string &str, char delim){
    std::vector<Type> ret;
    std::istringstream ss(str);
    std::string tmp;
    while(std::getline(ss, tmp, delim)){
        ret.push_back(boost::lexical_cast<Type>(tmp));
    }
    return ret;
}

template<>
inline std::vector<std::string> split<std::string>(const std::string &s, char delim){
    std::vector<std::string> ret;
    std::istringstream ss(s);
    ret.push_back(std::string());
    while(std::getline(ss, ret.back(), delim)){
        ret.push_back(std::string());
    }
    ret.pop_back();
    return ret;
}

boost::uint64_t getUtcTime();		// 单位毫秒。
boost::uint64_t getLocalTime();		// 单位毫秒。
boost::uint64_t getUtcTimeFromLocal(boost::uint64_t local);
boost::uint64_t getLocalTimeFromUtc(boost::uint64_t utc);

boost::uint64_t getMonoClock();		// 单位微秒。

// 在区间 [lower, upper) 范围内生成伪随机数。
// 前置条件：lower < upper
boost::uint32_t rand32();
boost::uint64_t rand64();
boost::uint32_t rand32(boost::uint32_t lower, boost::uint32_t upper);
double randDouble(double lower = 0.0, double upper = 1.0);

boost::shared_ptr<const char> getErrorDesc(int errCode = errno) throw();

}

#endif
