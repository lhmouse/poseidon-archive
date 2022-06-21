// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SOCKET_TCP_SERVER_SOCKET_
#define POSEIDON_SOCKET_TCP_SERVER_SOCKET_

#include "../fwd.hpp"
#include "tcp_socket.hpp"

namespace poseidon {

class TCP_Server_Socket
  : public TCP_Socket
  {
  private:
    // This class adds no data member.

  protected:
    // Accepts a connected socket.
    explicit
    TCP_Server_Socket(unique_posix_fd&& fd);

  protected:
    // This function implements `Abstract_Socket`.
    // The default implementation prints a message.
    virtual
    void
    do_abstract_socket_on_closed(int err) override;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(TCP_Server_Socket);

    // This class adds no public function.
  };

}  // namespace poseidon


#endif
