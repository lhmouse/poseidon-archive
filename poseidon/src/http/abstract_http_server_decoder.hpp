// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_ABSTRACT_HTTP_SERVER_DECODER_HPP_
#define POSEIDON_HTTP_ABSTRACT_HTTP_SERVER_DECODER_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_HTTP_Server_Decoder
  : public ::asteria::Rcfwd<Abstract_HTTP_Server_Decoder>
  {
  public:

  public:
    Abstract_HTTP_Server_Decoder()
      noexcept
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_HTTP_Server_Decoder);
  };

}  // namespace poseidon

#endif
