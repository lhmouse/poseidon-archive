// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SSL_FACTORIES_HPP_
#define POSEIDON_SSL_FACTORIES_HPP_

#include "cxx_ver.hpp"
#include "ssl_raii.hpp"
#include <boost/scoped_ptr.hpp>

namespace Poseidon {

extern void init_ssl_once();

class Ssl_filter;

class Ssl_server_factory {
private:
	const Unique_ssl_ctx m_ssl_ctx;

public:
	explicit Ssl_server_factory(const char *certificate, const char *private_key);
	~Ssl_server_factory();

public:
	void create_ssl_filter(boost::scoped_ptr<Ssl_filter> &ssl_filter, int fd);
};

class Ssl_client_factory {
private:
	const Unique_ssl_ctx m_ssl_ctx;

public:
	explicit Ssl_client_factory(bool verify_peer);
	~Ssl_client_factory();

public:
	void create_ssl_filter(boost::scoped_ptr<Ssl_filter> &ssl_filter, int fd);
};

}

#endif
