// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SSL_FACTORIES_HPP_
#define POSEIDON_SSL_FACTORIES_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "ssl_raii.hpp"

namespace Poseidon {

class SslFactoryBase : NONCOPYABLE {
private:
	const UniqueSslCtx m_sslCtx;

public:
	explicit SslFactoryBase(UniqueSslCtx sslCtx);
	virtual ~SslFactoryBase() = 0;

public:
	UniqueSsl createSsl() const;
};

class ServerSslFactory : public SslFactoryBase {
public:
	ServerSslFactory(const char *cert, const char *privateKey);
	~ServerSslFactory();
};

class ClientSslFactory : public SslFactoryBase {
public:
	ClientSslFactory();
	~ClientSslFactory();
};

}

#endif
