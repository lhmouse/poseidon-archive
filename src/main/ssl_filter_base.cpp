// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "ssl_filter_base.hpp"
#include <errno.h>
#include <openssl/ssl.h>
#include "exception.hpp"
#include "log.hpp"

namespace Poseidon {

namespace {
	void setErrnoBySslRet(::SSL *ssl, int ret){
		const int err = ::SSL_get_error(ssl, ret);
		LOG_POSEIDON_DEBUG("SSL ret = ", ret, ", error = ", err);
		switch(err){
		case SSL_ERROR_NONE:
		case SSL_ERROR_ZERO_RETURN:
			errno = 0;
			break;

		case SSL_ERROR_WANT_READ:
		case SSL_ERROR_WANT_WRITE:
		case SSL_ERROR_WANT_CONNECT:
		case SSL_ERROR_WANT_ACCEPT:
			errno = EAGAIN;
			break;

		case SSL_ERROR_SYSCALL:
			break;

		default:
			errno = EPERM;
			break;
		}
	}
}

SslFilterBase::SslFilterBase(Move<UniqueSsl> ssl, int fd)
	: m_ssl(STD_MOVE(ssl)), m_fd(fd)
	, m_established(false)
{
	if(!::SSL_set_fd(m_ssl.get(), fd)){
		DEBUG_THROW(Exception, SSLIT("::SSL_set_fd() failed"));
	}
}
SslFilterBase::~SslFilterBase(){
	if(m_established){
		LOG_POSEIDON_DEBUG("Shutting down SSL...");
		::SSL_shutdown(m_ssl.get());
	}
}

long SslFilterBase::read(void *data, unsigned long size){
	if(!m_established){
		if(!establish()){
			LOG_POSEIDON_DEBUG("Waiting for SSL handshake...");
			errno = EAGAIN;
			return -1;
		}
		m_established = true;
	}
	const long ret = ::SSL_read(m_ssl.get(), data, size);
	if(ret < 0){
		setErrnoBySslRet(m_ssl.get(), ret);
	}
	return ret;
}
long SslFilterBase::write(const void *data, unsigned long size){
	if(!m_established){
		if(!establish()){
			LOG_POSEIDON_DEBUG("Waiting for SSL handshake...");
			errno = EAGAIN;
			return -1;
		}
		m_established = true;
	}
	const long ret = ::SSL_write(m_ssl.get(), data, size);
	if(ret < 0){
		setErrnoBySslRet(m_ssl.get(), ret);
	}
	return ret;
}

}
