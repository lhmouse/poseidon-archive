// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SSL_FACTORIES_HPP_
#define POSEIDON_SSL_FACTORIES_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "ssl_raii.hpp"
#include <boost/scoped_ptr.hpp>

namespace Poseidon {

extern void init_ssl_once();

class SslFilter;

class SslServerFactory : NONCOPYABLE {
private:
	const UniqueSslCtx m_ssl_ctx;

public:
	explicit SslServerFactory(const char *certificate, const char *private_key);
	~SslServerFactory();

public:
	void create_ssl_filter(boost::scoped_ptr<SslFilter> &ssl_filter, int fd);
};

class SslClientFactory : NONCOPYABLE {
private:
	const UniqueSslCtx m_ssl_ctx;

public:
	explicit SslClientFactory(bool verify_peer);
	~SslClientFactory();

public:
	void create_ssl_filter(boost::scoped_ptr<SslFilter> &ssl_filter, int fd);
};

}

#endif
