// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_RANDOM_HPP_
#define POSEIDON_RANDOM_HPP_

#include <boost/cstdint.hpp>

namespace Poseidon {

// 在区间 [lower, upper) 范围内生成伪随机数。
// 前置条件：lower < upper
extern boost::uint32_t rand32();
extern boost::uint64_t rand64();
extern boost::uint32_t rand32(boost::uint32_t lower, boost::uint32_t upper);
extern double rand_double(double lower = 0.0, double upper = 1.0);

}

#endif
