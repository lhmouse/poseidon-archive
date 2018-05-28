// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SYSTEM_HTTP_SERVLET_BASE_HPP_
#define POSEIDON_SYSTEM_HTTP_SERVLET_BASE_HPP_

#include <boost/weak_ptr.hpp>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class Json_object;

class System_http_servlet_base {
public:
	System_http_servlet_base() noexcept = default;
	virtual ~System_http_servlet_base();

	System_http_servlet_base(const System_http_servlet_base &) = delete;
	System_http_servlet_base &operator=(const System_http_servlet_base &) = delete;

public:
	virtual const char * get_uri() const = 0;
	virtual void handle_get(Json_object &response) const = 0;
	virtual void handle_post(Json_object &response, Json_object request) const = 0;
};

}

#endif
