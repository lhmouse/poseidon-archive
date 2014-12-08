// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SSL_FILTER_BASE_HPP_
#define POSEIDON_SSL_FILTER_BASE_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "ssl_raii.hpp"

namespace Poseidon {

class SslFilterBase : NONCOPYABLE {
private:
	const UniqueSsl m_ssl;
	const int m_fd;

	bool m_established;

public:
	SslFilterBase(Move<UniqueSsl> ssl, int fd);
	virtual ~SslFilterBase();

protected:
	virtual bool establish() = 0;

public:
	::SSL *getSsl() const {
		return m_ssl.get();
	}

	long read(void *data, unsigned long size);
	long write(const void *data, unsigned long size);
};

}

#endif
