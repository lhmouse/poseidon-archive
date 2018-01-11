// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SSL_FILTER_HPP_
#define POSEIDON_SSL_FILTER_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "ssl_raii.hpp"
#include "mutex.hpp"

namespace Poseidon {

class SslFilter : NONCOPYABLE {
public:
	enum Direction {
		DIR_UNSPECIFIED = 0,
		DIR_TO_CONNECT  = 1,
		DIR_TO_ACCEPT   = 2,
	};

private:
	const UniqueSsl m_ssl;

	mutable Mutex m_mutex;

public:
	SslFilter(Move<UniqueSsl> ssl, Direction dir, int fd);
	virtual ~SslFilter();

public:
	long recv(void *data, unsigned long size);
	long send(const void *data, unsigned long size);
	void send_fin() NOEXCEPT;
};

}

#endif
