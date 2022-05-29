// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.ipp"
#include "../src/socket/abstract_tcp_server_socket.hpp"
#include "../src/socket/abstract_tcp_socket.hpp"
#include "../src/static/network_driver.hpp"
#include "../src/utils.hpp"

namespace {
using namespace poseidon;

struct Example_Session : Abstract_TCP_Socket
  {
    explicit
    Example_Session(unique_FD&& fd)
      : Abstract_TCP_Socket(::std::move(fd))
      { }

    void
    do_socket_on_receive(linear_buffer& rqueue) override
      {
        cow_string str(rqueue.begin(), rqueue.end());
        rqueue.clear();

        POSEIDON_LOG_WARN("example TCP session received: $1", str);
        this->do_socket_send(str);
      }
  };

rcptr<Abstract_TCP_Socket> s_client;  // only one client is allowed

constexpr char bind[] = "0.0.0.0";
constexpr uint16_t port = 3809;

struct Example_Server : Abstract_TCP_Server_Socket
  {
    explicit
    Example_Server()
      : Abstract_TCP_Server_Socket(bind, port)
      {
        POSEIDON_LOG_WARN("example TCP server listening: $1", this->get_local_address());
      }

    uptr<Abstract_TCP_Socket>
    do_socket_on_accept_tcp(unique_FD&& fd, const Socket_Address& addr) override
      {
        POSEIDON_LOG_WARN("example TCP server accepted client: $1", addr);

        return ::rocket::make_unique<Example_Session>(::std::move(fd));
      }

    void
    do_socket_on_register_tcp(rcptr<Abstract_TCP_Socket>&& sock) override
      {
        s_client = ::std::move(sock);
      }
  };

const auto s_server = Network_Driver::insert(::rocket::make_unique<Example_Server>());

}  // namespace
