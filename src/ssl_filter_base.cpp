// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "ssl_filter_base.hpp"
#include <errno.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include "exception.hpp"
#include "log.hpp"

namespace Poseidon {

namespace {
	void dump_error_queue(){
		unsigned long e;
		while((e = ::ERR_get_error()) != 0){
			char str[1024];
			::ERR_error_string_n(e, str, sizeof(str));
			LOG_POSEIDON_WARNING("SSL error: ", str);
		}
	}
	void set_errno_by_ssl_ret(::SSL *ssl, int ret){
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
{
	if(!::SSL_set_fd(m_ssl.get(), fd)){
		DEBUG_THROW(Exception, sslit("::SSL_set_fd() failed"));
	}
}
SslFilterBase::~SslFilterBase(){
	::SSL_shutdown(m_ssl.get());
}

long SslFilterBase::read(void *data, unsigned long size){
	dump_error_queue();
	const long ret = ::SSL_read(m_ssl.get(), data, size);
	if(ret < 0){
		set_errno_by_ssl_ret(m_ssl.get(), ret);
	}
	return ret;
}
long SslFilterBase::write(const void *data, unsigned long size){
	dump_error_queue();
	const long ret = ::SSL_write(m_ssl.get(), data, size);
	if(ret < 0){
		set_errno_by_ssl_ret(m_ssl.get(), ret);
	}
	return ret;
}

}
