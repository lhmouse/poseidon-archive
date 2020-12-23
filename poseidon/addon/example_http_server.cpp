// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.hpp"
#include "../src/socket/abstract_tcp_server_socket.hpp"
#include "../src/socket/abstract_tcp_socket.hpp"
#include "../src/static/network_driver.hpp"
#include "../src/utils.hpp"

#include "../src/http/enums.hpp"
#include "../src/http/option_map.hpp"
#include "../src/http/abstract_http_server_decoder.hpp"

namespace {
using namespace poseidon;

struct Example_Session : Abstract_TCP_Socket, Abstract_HTTP_Server_Decoder
  {
    explicit
    Example_Session(unique_FD&& fd)
      : Abstract_TCP_Socket(::std::move(fd))
      { }

    void
    do_socket_on_receive(char* data, size_t size)
      override
      {
        this->http_server_decode_stream(data, size);
      }

    void
    do_socket_on_close(int /*err*/)
      override
      {
        this->http_server_decode_end_of_stream();
      }

    void
    do_http_server_on_headers(HTTP_Method meth, cow_string&& target, HTTP_Version ver,
                              Option_Map&& headers)
      override
      {
        POSEIDON_LOG_FATAL("method:  $1\n"
                           "target:  $2\n"
                           "version: $3\n"
                           "headers:\n"
                           "$4",
                           format_http_method(meth),
                           target,
                           format_http_version(ver),
                           headers);
      }

    void
    do_http_server_on_entity(uint64_t offset, char* data, size_t size)
      override
      {
        POSEIDON_LOG_ERROR("entity (offset $1):\n"
                           "$2",
                           offset,
                           cow_string(data, size));
      }

    void
    do_http_server_on_end_of_entity()
      override
      {
        POSEIDON_LOG_ERROR("end of entity");
      }

    void
    do_http_server_on_tunnel_data(char* data, size_t size)
      override
      {
        POSEIDON_LOG_INFO("tunnel:\n"
                          "$1",
                          cow_string(data, size));
      }

    void
    do_http_server_on_tunnel_closure()
      override
      {
        POSEIDON_LOG_INFO("tunnel closure");
      }

    void
    do_http_server_on_websocket_frame(WebSocket_Opcode opcode, char* data, size_t size)
      override
      {
        POSEIDON_LOG_WARN("websocket (opcode $1):\n"
                          "$2",
                          opcode, cow_string(data, size));
      }

    void
    do_http_server_on_websocket_closure(WebSocket_Status stat, char* data, size_t size)
      override
      {
        POSEIDON_LOG_WARN("websocket closure: $1 ($2)", stat, cow_string(data, size));
      }

    bool
    do_http_server_close()
      override
      {
        return this->close();
      }
  };

rcptr<Abstract_TCP_Socket> s_client;  // only one client is allowed

constexpr char bind[] = "0.0.0.0";
constexpr uint16_t port = 3806;

struct Example_Server : Abstract_TCP_Server_Socket
  {
    explicit
    Example_Server()
      : Abstract_TCP_Server_Socket(bind, port)
      {
        POSEIDON_LOG_WARN("example HTTP server listening: $1",
                          this->get_local_address());
      }

    uptr<Abstract_TCP_Socket>
    do_socket_on_accept_tcp(unique_FD&& fd, const Socket_Address& addr)
      override
      {
        POSEIDON_LOG_WARN("example HTTP server accepted client: $1", addr);

        return ::rocket::make_unique<Example_Session>(::std::move(fd));
      }

    void
    do_socket_on_register_tcp(rcptr<Abstract_TCP_Socket>&& sock)
      override
      {
        s_client = ::std::move(sock);
      }
  };

const auto s_server = Network_Driver::insert(::rocket::make_unique<Example_Server>());

}  // namespace
