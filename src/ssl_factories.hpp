// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SSL_FACTORIES_HPP_
#define POSEIDON_SSL_FACTORIES_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "ssl_raii.hpp"

namespace Poseidon {

class SslFactoryBase : NONCOPYABLE {
protected:
	UniqueSslCtx m_ctx;

protected:
	explicit SslFactoryBase();
	virtual ~SslFactoryBase() = 0;

public:
	void create_ssl(UniqueSsl &ssl) const;
};

class ServerSslFactory : public SslFactoryBase {
public:
	explicit ServerSslFactory(const char *cert, const char *private_key);
	~ServerSslFactory() OVERRIDE;
};

class ClientSslFactory : public SslFactoryBase {
public:
	explicit ClientSslFactory(bool verify_peer);
	~ClientSslFactory() OVERRIDE;
};

}

#endif
