// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../poseidon/precompiled.ipp"
#include "../poseidon/socket/listen_socket.hpp"
#include "../poseidon/socket/ssl_socket.hpp"
#include "../poseidon/static/network_driver.hpp"
#include "../poseidon/static/async_logger.hpp"
#include "../poseidon/utils.hpp"

namespace {
using namespace poseidon;

const Socket_Address listen_address(::rocket::sref("[::]:3808"));

struct Example_Session : SSL_Socket
  {
    explicit
    Example_Session(unique_posix_fd&& fd, const SSL_CTX_ptr& ssl_ctx)
      : SSL_Socket(::std::move(fd), ssl_ctx)
      {
      }

    charbuf_256
    do_on_ssl_alpn_request(cow_vector<charbuf_256>&& protos) override
      {
        for(const auto& proto : protos)
          POSEIDON_LOG_ERROR(("ALPN: received `$1`"), proto);

        // select no protocol
        return {};
      }

    void
    do_on_ssl_stream(linear_buffer& data) override
      {
        cow_string str(data.begin(), data.end());
        data.clear();
        POSEIDON_LOG_WARN(("example SSL server received from `$1`: $2"), this->remote_address(), str);
        this->ssl_send(str);
      }
  };

struct Example_Server : Listen_Socket
  {
    shared_ptr<Example_Session> m_client;

    explicit
    Example_Server()
      : Listen_Socket(listen_address)
      {
        POSEIDON_LOG_WARN(("example SSL server listening on `$1`"), this->local_address());
      }

    shared_ptr<Abstract_Socket>
    do_on_listen_new_client_opt(unique_posix_fd&& fd) override
      {
        recursive_mutex::unique_lock io_lock;
        auto& driver = this->do_abstract_socket_lock_driver(io_lock);

        this->m_client = ::std::make_shared<Example_Session>(::std::move(fd), driver.default_server_ssl_ctx());
        POSEIDON_LOG_WARN(("example SSL server accepted connection from `$1`"), this->m_client->remote_address());
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
