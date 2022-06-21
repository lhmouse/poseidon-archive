// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_UDP_SERVER_SOCKET_
#define POSEIDON_SOCKET_UDP_SERVER_SOCKET_

#include "../fwd.hpp"
#include "udp_socket.hpp"

namespace poseidon {

class UDP_Server_Socket
  : public UDP_Socket
  {
  private:
    // This class adds no data member.

  protected:
    // Creates a socket that is bound onto the given address.
    explicit
    UDP_Server_Socket(const Socket_Address& addr);

  protected:
    // This function implements `Abstract_Socket`.
    // The default implementation prints a message.
    virtual
    void
    do_abstract_socket_on_closed(int err) override;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(UDP_Server_Socket);

    // This class adds no public function.
  };

}  // namespace poseidon


#endif
