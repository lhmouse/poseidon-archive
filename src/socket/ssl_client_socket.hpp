// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_SSL_CLIENT_SOCKET_
#define POSEIDON_SOCKET_SSL_CLIENT_SOCKET_

#include "../fwd.hpp"
#include "ssl_socket.hpp"

namespace poseidon {

class SSL_Client_Socket
  : public SSL_Socket
  {
  private:
    // This class adds no data member.

  protected:
    // Creates a socket that connects to the given address.
    explicit
    SSL_Client_Socket(const Socket_Address& addr, const SSL_CTX_ptr& ssl_ctx);

  protected:
    // This function implements `Abstract_Socket`.
    // The default implementation prints a message.
    virtual
    void
    do_abstract_socket_on_closed(int err) override;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(SSL_Client_Socket);

    // This class adds no public function.
  };

}  // namespace poseidon


#endif
