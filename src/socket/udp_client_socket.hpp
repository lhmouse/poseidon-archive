// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_UDP_CLIENT_SOCKET_
#define POSEIDON_SOCKET_UDP_CLIENT_SOCKET_

#include "../fwd.hpp"
#include "udp_socket.hpp"

namespace poseidon {

class UDP_Client_Socket
  : public UDP_Socket
  {
  private:
    // This class adds no data member.

  protected:
    // Creates a socket for sending data.
    explicit
    UDP_Client_Socket();

  protected:
    // This function implements `Abstract_Socket`.
    // The default implementation prints a message.
    virtual
    void
    do_abstract_socket_on_closed(int err) override;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(UDP_Client_Socket);

    // This class adds no public function.
  };

}  // namespace poseidon


#endif
