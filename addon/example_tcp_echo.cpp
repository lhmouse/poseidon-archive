// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../poseidon/precompiled.ipp"
#include "../poseidon/socket/listen_socket.hpp"
#include "../poseidon/socket/tcp_socket.hpp"
#include "../poseidon/static/network_driver.hpp"
#include "../poseidon/static/async_logger.hpp"
#include "../poseidon/utils.hpp"

namespace {
using namespace poseidon;

constexpr char bind[] = "[::]";
constexpr uint16_t port = 3809;

struct Example_Session : TCP_Socket
  {
    explicit
    Example_Session(unique_posix_fd&& fd)
      :
        TCP_Socket(::std::move(fd))
      {
      }

    void
    do_on_tcp_stream(linear_buffer& data) override
      {
        cow_string str(data.begin(), data.end());
        data.clear();
        POSEIDON_LOG_WARN(("example TCP server received from `$1`: $2"), this->get_remote_address(), str);
        this->tcp_send(str);
      }
  };

struct Example_Server : Listen_Socket
  {
    shared_ptr<Example_Session> m_client;

    explicit
    Example_Server()
      :
        Listen_Socket(Socket_Address(::rocket::sref(bind), port))
      {
        POSEIDON_LOG_WARN(("example TCP server listening on `$1`"), this->get_local_address());
      }

    shared_ptr<Abstract_Socket>
    do_on_listen_new_client_opt(unique_posix_fd&& fd) override
      {
        this->m_client = ::std::make_shared<Example_Session>(::std::move(fd));
        POSEIDON_LOG_WARN(("example TCP server accepted connection from `$1`"), this->m_client->get_remote_address());
        return this->m_client;
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
