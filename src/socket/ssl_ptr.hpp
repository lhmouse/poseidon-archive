// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_SSL_PTR_
#define POSEIDON_SOCKET_SSL_PTR_

#include "../fwd.hpp"
#include <openssl/ssl.h>

namespace poseidon {

class SSL_ptr
  {
  private:
    ::SSL* m_ptr;

  public:
    constexpr
    SSL_ptr(::SSL* ptr = nullptr) noexcept
      : m_ptr(ptr)
      { }

    SSL_ptr(const SSL_ptr& other) noexcept
      : m_ptr(other.do_up_ref())
      { }

    SSL_ptr(SSL_ptr&& other) noexcept
      : m_ptr(other.release())
      { }

    SSL_ptr&
    operator=(const SSL_ptr& other) & noexcept
      {
        this->reset(other.do_up_ref());
        return *this;
      }

    SSL_ptr&
    operator=(SSL_ptr&& other) & noexcept
      {
        this->reset(other.release());
        return *this;
      }

    ~SSL_ptr()
      {
        if(this->m_ptr)
          ::SSL_free(this->m_ptr);
      }

  private:
    ::SSL*
    do_up_ref() const noexcept
      {
        auto ptr_old = this->m_ptr;
        if(ptr_old)
          ::SSL_up_ref(ptr_old);
        return ptr_old;
      }

  public:
    explicit constexpr operator
    bool() const noexcept
      { return this->m_ptr != nullptr;  }

    constexpr operator
    ::SSL*() const noexcept
      { return this->m_ptr;  }

    constexpr
    ::SSL*
    get() const noexcept
      { return this->m_ptr;  }

    ::SSL*
    release() noexcept
      {
        return ::std::exchange(this->m_ptr, nullptr);
      }

    SSL_ptr&
    reset(::SSL* ptr_new = nullptr) noexcept
      {
        auto ptr_old = ::std::exchange(this->m_ptr, ptr_new);
        if(ptr_old)
          ::SSL_free(ptr_old);
        return *this;
      }

    SSL_ptr&
    swap(SSL_ptr& other) noexcept
      {
        ::std::swap(this->m_ptr, other.m_ptr);
        return *this;
      }
  };

inline
void
swap(SSL_ptr& lhs, SSL_ptr& rhs) noexcept
  {
    lhs.swap(rhs);
  }

}  // namespace poseidon

#endif
