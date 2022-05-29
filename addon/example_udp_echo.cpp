// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.ipp"
#include "../src/socket/abstract_udp_server_socket.hpp"
#include "../src/static/network_driver.hpp"
#include "../src/utils.hpp"

namespace {
using namespace poseidon;

constexpr char bind[] = "0.0.0.0";
constexpr uint16_t port = 3807;

struct Example_Server : Abstract_UDP_Server_Socket
  {
    explicit
    Example_Server()
      : Abstract_UDP_Server_Socket(bind, port)
      {
        POSEIDON_LOG_WARN("example UDP server listening: $1", this->get_local_address());
      }

    void
    do_socket_on_packet(const Socket_Address& addr, linear_buffer&& rqueue) override
      {
        cow_string str(rqueue.begin(), rqueue.end());
        rqueue.clear();

        POSEIDON_LOG_WARN("example UDP session received: $1", str);
        this->do_socket_send(addr, str);
      }
  };

const auto s_server = Network_Driver::insert(::rocket::make_unique<Example_Server>());

}  // namespace
