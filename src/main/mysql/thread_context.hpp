// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_THREAD_CONTEXT_HPP_
#define POSEIDON_MYSQL_THREAD_CONTEXT_HPP_

#include "../cxx_ver.hpp"
#include <cstddef>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>

namespace Poseidon {

class MySqlThreadContext : boost::noncopyable {
private:
	void *operator new(std::size_t);
	void operator delete(void *) NOEXCEPT;
	void *operator new[](std::size_t);
	void operator delete[](void *) NOEXCEPT;

public:
	MySqlThreadContext();
	~MySqlThreadContext();
};

}

#endif
