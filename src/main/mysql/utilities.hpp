// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_UTILITIES_HPP_
#define POSEIDON_MYSQL_UTILITIES_HPP_

#include <iosfwd>
#include <string>
#include <boost/cstdint.hpp>

namespace Poseidon {

extern void quoteStringForSql(std::ostream &os, const std::string &str);

inline double datetimeFromTime(boost::uint64_t ms){
	return static_cast<double>(ms) / (24 * 3600 * 1000);
}
inline boost::uint64_t timeFromDateTime(double datetime){
	return static_cast<boost::uint64_t>(datetime * (24 * 3600 * 1000));
}

extern void formatDateTime(std::ostream &os, double datetime);
extern double scanDateTime(const char *str);

}

#endif
