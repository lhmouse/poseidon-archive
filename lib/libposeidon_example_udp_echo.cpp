// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../poseidon/precompiled.ipp"
#include "../poseidon/socket/udp_socket.hpp"
#include "../poseidon/static/network_driver.hpp"
#include "../poseidon/static/async_logger.hpp"
#include "../poseidon/utils.hpp"

namespace {
using namespace poseidon;

const Socket_Address listen_address(::rocket::sref("[::]:3807"));

struct Example_Server : UDP_Socket
  {
    explicit
    Example_Server()
      : UDP_Socket(listen_address)
      {
        POSEIDON_LOG_WARN(("example UDP server listening on `$1`"), this->get_local_address());
      }

    void
    do_on_udp_packet(Socket_Address&& saddr, linear_buffer&& data) override
      {
        cow_string str(data.begin(), data.end());
        data.clear();
        POSEIDON_LOG_WARN(("example UDP server received from `$1`: $2"), saddr, str);
        this->udp_send(saddr, str);
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
