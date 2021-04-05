// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.hpp"
#include "../src/socket/abstract_tls_server_socket.hpp"
#include "../src/socket/abstract_tls_socket.hpp"
#include "../src/static/network_driver.hpp"
#include "../src/utils.hpp"

namespace {
using namespace poseidon;

struct Example_Session : Abstract_TLS_Socket
  {
    explicit
    Example_Session(unique_FD&& fd, const OpenSSL_Context& ctx)
      : Abstract_TLS_Socket(::std::move(fd), ctx)
      { }

    void
    do_socket_on_receive(char* data, size_t size) override
      {
        POSEIDON_LOG_WARN("example TLS session received: $1", cow_string(data, size));
        this->do_socket_send(data, size);
      }
  };

rcptr<Abstract_TLS_Socket> s_client;  // only one client is allowed

constexpr char bind[] = "0.0.0.0";
constexpr uint16_t port = 3808;

struct Example_Server : Abstract_TLS_Server_Socket
  {
    explicit
    Example_Server()
      : Abstract_TLS_Server_Socket(bind, port)
      {
        POSEIDON_LOG_WARN("example TLS server listening: $1",
                          this->get_local_address());
      }

    uptr<Abstract_TLS_Socket>
    do_socket_on_accept_tls(unique_FD&& fd, const Socket_Address& addr) override
      {
        POSEIDON_LOG_WARN("example TLS server accepted client: $1", addr);

        return ::rocket::make_unique<Example_Session>(::std::move(fd), *this);
      }

    void
    do_socket_on_register_tls(rcptr<Abstract_TLS_Socket>&& sock) override
      {
        s_client = ::std::move(sock);
      }
  };

const auto s_server = Network_Driver::insert(::rocket::make_unique<Example_Server>());

}  // namespace
