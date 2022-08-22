// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.ipp"
#include "../src/socket/udp_socket.hpp"
#include "../src/static/network_driver.hpp"
#include "../src/static/async_logger.hpp"
#include "../src/utils.hpp"

namespace {
using namespace poseidon;

constexpr char bind[] = "[::]";
constexpr uint16_t port = 3807;

struct Example_Server : UDP_Socket
  {
    explicit
    Example_Server()
      : UDP_Socket(Socket_Address(::rocket::sref(bind), port))
      {
        POSEIDON_LOG_WARN(("example UDP server listening on `$1`"), this->get_local_address());
      }

    void
    do_on_udp_packet(Socket_Address&& addr, linear_buffer&& data) override
      {
        cow_string str(data.begin(), data.end());
        data.clear();
        POSEIDON_LOG_WARN(("example UDP server received from `$1`: $2"), addr, str);
        this->udp_send(addr, str);
      }
  };

shared_ptr<Example_Server>
do_create_server()
  {
    auto server = ::std::make_shared<Example_Server>();
    network_driver.insert(server);
    return server;
  }

const auto server = do_create_server();

}  // namespace
