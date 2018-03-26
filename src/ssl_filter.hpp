// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SSL_FILTER_HPP_
#define POSEIDON_SSL_FILTER_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "ssl_raii.hpp"
#include "mutex.hpp"

namespace Poseidon {

class Ssl_filter : NONCOPYABLE {
public:
	enum Direction {
		to_connect  = 1,
		to_accept   = 2,
	};

private:
	const Unique_ssl m_ssl;

	mutable Mutex m_mutex;

public:
	Ssl_filter(Move<Unique_ssl> ssl, Direction dir, int fd);
	virtual ~Ssl_filter();

public:
	long recv(void *data, unsigned long size);
	long send(const void *data, unsigned long size);
	void send_fin() NOEXCEPT;
};

}

#endif
