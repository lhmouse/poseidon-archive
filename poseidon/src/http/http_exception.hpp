// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_HTTP_EXCEPTION_HPP_
#define POSEIDON_HTTP_HTTP_EXCEPTION_HPP_

#include "../fwd.hpp"
#include <exception>

namespace poseidon {

class HTTP_Exception
  : public virtual exception
  {
  private:
    HTTP_Status m_status;
    cow_string m_desc;

  public:
    explicit
    HTTP_Exception(HTTP_Status status, cow_string&& desc) noexcept
      : m_status(status), m_desc(::std::move(desc))
      { }

  public:
    ASTERIA_COPYABLE_DESTRUCTOR(HTTP_Exception);

    HTTP_Status
    status() const noexcept
      { return this->m_status;  }

    const char*
    what() const noexcept override
      { return this->m_desc.c_str();  }
  };

// Composes a string and throws an exception.
#define POSEIDON_HTTP_THROW(status, ...)  \
    (throw ::poseidon::HTTP_Exception(status,  \
                 ::asteria::format_string("" __VA_ARGS__)))

}  // namespace asteria

#endif
