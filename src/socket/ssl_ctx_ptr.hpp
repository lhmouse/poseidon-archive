// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_SSL_CTX_PTR_
#define POSEIDON_SOCKET_SSL_CTX_PTR_

#include "../fwd.hpp"
#include <openssl/ssl.h>

namespace poseidon {

class SSL_CTX_ptr
  {
  private:
    ::SSL_CTX* m_ptr;

  public:
    constexpr
    SSL_CTX_ptr(::SSL_CTX* ptr = nullptr) noexcept
      : m_ptr(ptr)
      { }

    SSL_CTX_ptr(const SSL_CTX_ptr& other) noexcept
      : m_ptr(other.do_up_ref())
      { }

    SSL_CTX_ptr(SSL_CTX_ptr&& other) noexcept
      : m_ptr(other.release())
      { }

    SSL_CTX_ptr&
    operator=(const SSL_CTX_ptr& other) & noexcept
      {
        this->reset(other.do_up_ref());
        return *this;
      }

    SSL_CTX_ptr&
    operator=(SSL_CTX_ptr&& other) & noexcept
      {
        this->reset(other.release());
        return *this;
      }

    ~SSL_CTX_ptr()
      {
        if(this->m_ptr)
          ::SSL_CTX_free(this->m_ptr);
      }

  private:
    ::SSL_CTX*
    do_up_ref() const noexcept
      {
        auto ptr_old = this->m_ptr;
        if(ptr_old)
          ::SSL_CTX_up_ref(ptr_old);
        return ptr_old;
      }

  public:
    explicit constexpr operator
    bool() const noexcept
      { return this->m_ptr != nullptr;  }

    constexpr operator
    ::SSL_CTX*() const noexcept
      { return this->m_ptr;  }

    constexpr
    ::SSL_CTX*
    get() const noexcept
      { return this->m_ptr;  }

    ::SSL_CTX*
    release() noexcept
      {
        return ::std::exchange(this->m_ptr, nullptr);
      }

    SSL_CTX_ptr&
    reset(::SSL_CTX* ptr_new = nullptr) noexcept
      {
        auto ptr_old = ::std::exchange(this->m_ptr, ptr_new);
        if(ptr_old)
          ::SSL_CTX_free(ptr_old);
        return *this;
      }

    SSL_CTX_ptr&
    swap(SSL_CTX_ptr& other) noexcept
      {
        ::std::swap(this->m_ptr, other.m_ptr);
        return *this;
      }
  };

inline
void
swap(SSL_CTX_ptr& lhs, SSL_CTX_ptr& rhs) noexcept
  {
    lhs.swap(rhs);
  }

}  // namespace poseidon
#endif
