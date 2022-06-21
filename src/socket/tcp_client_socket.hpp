// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_TCP_CLIENT_SOCKET_
#define POSEIDON_SOCKET_TCP_CLIENT_SOCKET_

#include "../fwd.hpp"
#include "tcp_socket.hpp"

namespace poseidon {

class TCP_Client_Socket
  : public TCP_Socket
  {
  private:
    // This class adds no data member.

  protected:
    // Creates a socket that connects to the given address.
    explicit
    TCP_Client_Socket(const Socket_Address& addr);

  protected:
    // This function implements `Abstract_Socket`.
    // The default implementation prints a message.
    virtual
    void
    do_abstract_socket_on_closed(int err) override;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(TCP_Client_Socket);

    // This class adds no public function.
  };

}  // namespace poseidon


#endif
