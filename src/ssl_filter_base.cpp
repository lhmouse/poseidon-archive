// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "ssl_filter_base.hpp"
#include <sys/socket.h>
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
			errno = EWOULDBLOCK;
			break;
		case SSL_ERROR_SYSCALL:
			break;
		case SSL_ERROR_SSL:
			dump_error_queue();
			errno = EPERM;
			break;
		default:
			LOG_POSEIDON_ERROR("Unknown SSL error: ", err);
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
}

long SslFilterBase::recv(void *data, unsigned long size){
	const Mutex::UniqueLock lock(m_mutex);
	dump_error_queue();
	int ret;
	ret = ::SSL_read(m_ssl.get(), data, size);
	if(::SSL_get_shutdown(m_ssl.get()) & SSL_RECEIVED_SHUTDOWN){
		::shutdown(::SSL_get_rfd(m_ssl.get()), SHUT_RD);
	}
	if(ret < 0){
		set_errno_by_ssl_ret(m_ssl.get(), ret);
	}
	return ret;
}
long SslFilterBase::send(const void *data, unsigned long size){
	const Mutex::UniqueLock lock(m_mutex);
	dump_error_queue();
	int ret;
	if(::SSL_get_shutdown(m_ssl.get()) & SSL_SENT_SHUTDOWN){
		ret = ::SSL_shutdown(m_ssl.get());
		if(ret == 1){
			::shutdown(::SSL_get_wfd(m_ssl.get()), SHUT_WR);
		}
		ret = 0;
	} else {
		ret = ::SSL_write(m_ssl.get(), data, size);
	}
	if(ret < 0){
		set_errno_by_ssl_ret(m_ssl.get(), ret);
	}
	return ret;
}
void SslFilterBase::send_fin() NOEXCEPT {
	const Mutex::UniqueLock lock(m_mutex);
	dump_error_queue();
	int ret = ::SSL_shutdown(m_ssl.get());
	if(ret == 1){
		::shutdown(::SSL_get_wfd(m_ssl.get()), SHUT_WR);
	}
	if(ret < 0){
		set_errno_by_ssl_ret(m_ssl.get(), ret);
	}
}

}
