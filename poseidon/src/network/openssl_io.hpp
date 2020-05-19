// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_OPENSSL_IO_HPP_
#define POSEIDON_NETWORK_OPENSSL_IO_HPP_

#include "abstract_tls_io.hpp"
#include <openssl/ssl.h>

namespace poseidon {

class OpenSSL_IO
final
  : public Abstract_TLS_IO
  {
  public:
    enum Method : uint8_t
      {
        method_generic  = 0,  // unspecified
        method_connect  = 1,  // client method
        method_accept   = 2,  // server method
      };

  private:
    uptr<::SSL, void (&)(::SSL*)> m_ssl;

  public:
    OpenSSL_IO(uptr<::SSL, void (&)(::SSL*)>&& ssl, Method method)
      : m_ssl(::std::move(ssl))
      { this->set_method(method);  }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(OpenSSL_IO);

  protected:
    // Prepares for I/O.
    // This must precede any I/O operations.
    void
    set_method(Method method)
    noexcept;

    // Reads some data, like `::read()`.
    IO_Result
    read(void* data, size_t size)
    override;

    // Writes some data, like `::write()`.
    IO_Result
    write(const void* data, size_t size)
    override;

    // Initiates normal shutdown of this stream.
    IO_Result
    shutdown()
    override;
  };

}  // namespace poseidon

#endif
