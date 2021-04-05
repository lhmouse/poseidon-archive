// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_WEBSOCKET_EXCEPTION_HPP_
#define POSEIDON_HTTP_WEBSOCKET_EXCEPTION_HPP_

#include "../fwd.hpp"
#include <exception>

namespace poseidon {

class WebSocket_Exception
  : public virtual exception
  {
  private:
    WebSocket_Status m_status;
    cow_string m_desc;

  public:
    explicit
    WebSocket_Exception(WebSocket_Status status, cow_string&& desc) noexcept
      : m_status(status), m_desc(::std::move(desc))
      { }

  public:
    ASTERIA_COPYABLE_DESTRUCTOR(WebSocket_Exception);

    WebSocket_Status
    status() const noexcept
      { return this->m_status;  }

    const char*
    what() const noexcept override
      { return this->m_desc.c_str();  }
  };

// Composes a string and throws an exception.
#define POSEIDON_WEBSOCKET_THROW(status, ...)  \
    (throw ::poseidon::WebSocket_Exception(status,  \
                 ::asteria::format_string("" __VA_ARGS__)))

}  // namespace asteria

#endif
