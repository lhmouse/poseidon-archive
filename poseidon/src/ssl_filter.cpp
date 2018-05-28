// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "ssl_filter.hpp"
#include "exception.hpp"
#include "log.hpp"
#include "profiler.hpp"
#include <sys/socket.h>
#include <openssl/ssl.h>
#include <openssl/err.h>

namespace Poseidon {

namespace {
	void dump_error_queue(){
		unsigned long e;
		while((e = ::ERR_get_error()) != 0){
			char str[1024];
			::ERR_error_string_n(e, str, sizeof(str));
			POSEIDON_LOG_WARNING("SSL error: ", str);
		}
	}

	int get_errno_from_ssl_ret(::SSL *ssl, int ssl_ret){
		const int err = ::SSL_get_error(ssl, ssl_ret);
		POSEIDON_LOG_TRACE("SSL error: ssl_ret = ", ssl_ret, ", err = ", err);
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
			POSEIDON_LOG_ERROR("Unknown SSL error: ", ssl_ret);
			return EPERM;
		}
	}
}

Ssl_filter::Ssl_filter(Move<Unique_ssl> ssl, Ssl_filter::Direction dir, int fd)
	: m_ssl(STD_MOVE(ssl))
{
	if(dir == to_connect){
		::SSL_set_connect_state(m_ssl.get());
	} else if(dir == to_accept){
		::SSL_set_accept_state(m_ssl.get());
	}
	POSEIDON_THROW_UNLESS(::SSL_set_fd(m_ssl.get(), fd), Exception, Rcnts::view("::SSL_set_fd() failed"));
}
Ssl_filter::~Ssl_filter(){
	//
}

long Ssl_filter::recv(void *data, unsigned long size){
	POSEIDON_PROFILE_ME;

	const std::lock_guard<std::mutex> lock(m_mutex);
	int bytes_read = ::SSL_read(m_ssl.get(), data, static_cast<int>(std::min<unsigned long>(size, INT_MAX)));
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
long Ssl_filter::send(const void *data, unsigned long size){
	POSEIDON_PROFILE_ME;

	const std::lock_guard<std::mutex> lock(m_mutex);
	int bytes_written = ::SSL_write(m_ssl.get(), data, static_cast<int>(std::min<unsigned long>(size, INT_MAX)));
	if(bytes_written <= 0){
		if(::SSL_get_shutdown(m_ssl.get()) & SSL_SENT_SHUTDOWN){
			const int status = ::SSL_shutdown(m_ssl.get());
			if(status == 1){
				::shutdown(::SSL_get_wfd(m_ssl.get()), SHUT_WR);
			}
			if(status < 0){
				POSEIDON_LOG_WARNING("::SSL_shutdown() failed: status = ", status);
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
void Ssl_filter::send_fin() NOEXCEPT {
	POSEIDON_PROFILE_ME;

	const std::lock_guard<std::mutex> lock(m_mutex);
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
