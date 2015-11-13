// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SSL_FACTORIES_HPP_
#define POSEIDON_SSL_FACTORIES_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "ssl_raii.hpp"

namespace Poseidon {

class SslFactoryBase : NONCOPYABLE {
private:
	const UniqueSslCtx m_ssl_ctx;

public:
	explicit SslFactoryBase(UniqueSslCtx ssl_ctx);
	virtual ~SslFactoryBase() = 0;

public:
	UniqueSsl create_ssl() const;
};

class ServerSslFactory : public SslFactoryBase {
public:
	ServerSslFactory(const char *cert, const char *private_key);
	~ServerSslFactory();
};

class ClientSslFactory : public SslFactoryBase {
public:
	ClientSslFactory();
	~ClientSslFactory();
};

}

#endif
