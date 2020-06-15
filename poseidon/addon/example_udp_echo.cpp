// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.hpp"
#include "../src/socket/abstract_udp_server_socket.hpp"
#include "../src/static/network_driver.hpp"
#include "../src/utilities.hpp"

namespace {

using namespace poseidon;

constexpr char bind[] = "0.0.0.0";
constexpr uint16_t port = 3807;

struct Example_Server : Abstract_UDP_Server_Socket
  {
    Example_Server()
      : Abstract_UDP_Server_Socket(bind, port)
      {
        POSEIDON_LOG_WARN("example UDP server listening: $1",
                          this->get_local_address());
      }

    void
    do_on_async_receive(Socket_Address&& addr, void* data, size_t size)
    override
      {
        POSEIDON_LOG_WARN("example UDP server received from '$1': $2",
                          addr, cow_string(static_cast<char*>(data), size));

        this->async_send(addr, data, size);
      }
  };

const auto s_server = Network_Driver::insert(::rocket::make_unique<Example_Server>());

}  // namespace
