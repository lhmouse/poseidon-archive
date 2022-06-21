// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.ipp"
#include "../src/socket/listen_socket.hpp"
#include "../src/socket/tcp_server_socket.hpp"
#include "../src/static/network_driver.hpp"
#include "../src/static/async_logger.hpp"
#include "../src/utils.hpp"

namespace {
using namespace poseidon;

constexpr char bind[] = "[::]";
constexpr uint16_t port = 3809;

struct Example_Server : Listen_Socket
  {
    explicit
    Example_Server()
      : Listen_Socket(Socket_Address(bind, port))
      {
        POSEIDON_LOG_WARN(("example TCP server listening on `$1`"), this->get_local_address());
      }

    shared_ptr<Abstract_Socket>
    do_on_new_client_opt(Socket_Address&& addr) override
      {
        POSEIDON_LOG_WARN(("example TCP server accepted connection from `$1`"), addr);
        return nullptr;
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
