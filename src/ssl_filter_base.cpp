// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "ssl_filter_base.hpp"
#include <sys/socket.h>
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
	int get_errno_from_ssl_ret(::SSL *ssl, int ssl_ret){
		const int err = ::SSL_get_error(ssl, ssl_ret);
		LOG_POSEIDON_TRACE("SSL error: ssl_ret = ", ssl_ret, ", err = ", err);
		switch(err){
		case SSL_ERROR_NONE:
		case SSL_ERROR_ZERO_RETURN:
			return 0;
		case SSL_ERROR_WANT_READ:
		case SSL_ERROR_WANT_WRITE:
		case SSL_ERROR_WANT_CONNECT:
		case SSL_ERROR_WANT_ACCEPT:
			return EWOULDBLOCK;
		case SSL_ERROR_SYSCALL:
			return EPIPE;
		case SSL_ERROR_SSL:
			dump_error_queue();
			return EPERM;
		default:
			LOG_POSEIDON_ERROR("Unknown SSL error: ", ssl_ret);
			return EPERM;
		}
	}
}

SslFilterBase::SslFilterBase(Move<UniqueSsl> ssl, int fd)
	: m_ssl(STD_MOVE(ssl)), m_fd(fd)
{
	DEBUG_THROW_UNLESS(::SSL_set_fd(m_ssl.get(), fd), Exception, sslit("::SSL_set_fd() failed"));
}
SslFilterBase::~SslFilterBase(){ }

long SslFilterBase::recv(void *data, unsigned long size){
	const Mutex::UniqueLock lock(m_mutex);
	long bytes_read = ::SSL_read(m_ssl.get(), data, size);
	if(bytes_read <= 0){
		if(::SSL_get_shutdown(m_ssl.get()) & SSL_RECEIVED_SHUTDOWN){
			::shutdown(::SSL_get_rfd(m_ssl.get()), SHUT_RD);
		}
	}
	if(bytes_read < 0){
		const int err = get_errno_from_ssl_ret(m_ssl.get(), bytes_read);
		if(err == 0){
			bytes_read = 0;
		}
		errno = err;
	}
	return bytes_read;
}
long SslFilterBase::send(const void *data, unsigned long size){
	const Mutex::UniqueLock lock(m_mutex);
	long bytes_written = ::SSL_write(m_ssl.get(), data, size);
	if(bytes_written <= 0){
		if(::SSL_get_shutdown(m_ssl.get()) & SSL_SENT_SHUTDOWN){
			const int status = ::SSL_shutdown(m_ssl.get());
			if(status == 1){
				::shutdown(::SSL_get_wfd(m_ssl.get()), SHUT_WR);
			}
			if(status < 0){
				LOG_POSEIDON_WARNING("::SSL_shutdown() failed: status = ", status);
			}
		}
	}
	if(bytes_written < 0){
		const int err = get_errno_from_ssl_ret(m_ssl.get(), bytes_written);
		if(err == 0){
			bytes_written = 0;
		}
		errno = err;
	}
	return bytes_written;
}
void SslFilterBase::send_fin() NOEXCEPT {
	const Mutex::UniqueLock lock(m_mutex);
	const int status = ::SSL_shutdown(m_ssl.get());
	if(status == 1){
		::shutdown(::SSL_get_wfd(m_ssl.get()), SHUT_WR);
	}
	if(status < 0){
		const int err = get_errno_from_ssl_ret(m_ssl.get(), status);
		errno = err;
	}
}

}
