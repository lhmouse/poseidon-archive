// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SYSTEM_SERVLET_BASE_HPP_
#define POSEIDON_SYSTEM_SERVLET_BASE_HPP_

#include "cxx_util.hpp"
#include <boost/weak_ptr.hpp>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class JsonObject;

class SystemServletBase : NONCOPYABLE {
public:
	virtual ~SystemServletBase();

public:
	virtual const char *get_uri() const = 0;
	virtual void handle_get(JsonObject &response) const = 0;
	virtual void handle_post(JsonObject &response, JsonObject request) const = 0;
};

}

#endif
