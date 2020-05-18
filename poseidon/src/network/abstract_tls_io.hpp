// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_NETWORK_ABSTRACT_TLS_IO_HPP_
#define POSEIDON_NETWORK_ABSTRACT_TLS_IO_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_TLS_IO
  : public virtual ::asteria::Rcbase
  {
  public:
    Abstract_TLS_IO()
    noexcept
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_TLS_IO);

  public:
    // Reads some data, like `::read()`.
    // This function shall return a negative value for restartable errors
    // such as `EINTR` and `EAGAIN`. An exception shall be thrown in case of
    // non-restartable errors.
    virtual
    ptrdiff_t
    read(void* data, size_t size)
      = 0;

    // Writes some data, like `::write()`.
    // This function shall return a negative value for restartable errors
    // such as `EINTR` and `EAGAIN`. An exception shall be thrown in case of
    // non-restartable errors.
    virtual
    ptrdiff_t
    write(const void* data, size_t size)
      = 0;

    // Initiates normal shutdown of this stream.
    // This function may be called repeatedly. Only after it returns `true`
    // shall the stream be considered to have been closed completely.
    virtual
    bool
    shutdown()
      = 0;
  };

}  // namespace poseidon

#endif
