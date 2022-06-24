// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../src/precompiled.ipp"
#include "../src/socket/ssl_client_socket.hpp"
#include "../src/static/network_driver.hpp"
#include "../src/static/async_logger.hpp"
#include "../src/utils.hpp"

namespace {
using namespace poseidon;

constexpr char conn[] = "93.184.216.34";  // example.org
constexpr uint16_t port = 443;

struct Example_Session : SSL_Client_Socket
  {
    explicit
    Example_Session()
      : SSL_Client_Socket(Socket_Address(conn, port), network_driver.default_client_ssl_ctx())
      {
        static constexpr char data[] =
            "GET / HTTP/1.1\r\n"
            "Host: example.org\r\n"
            "Connection: close\r\n"
            "\r\n";

        this->ssl_send(data, ::strlen(data));
      }

    void
    do_on_ssl_stream(linear_buffer& data) override
      {
        cow_string str(data.begin(), data.end());
        data.clear();
        POSEIDON_LOG_WARN(("example SSL client received from `$1`:\n\n$2"), this->get_remote_address(), str);
      }
  };

shared_ptr<Example_Session>
do_create_client()
  {
    auto client = ::std::make_shared<Example_Session>();
    network_driver.insert(client);
    return client;
  }

const auto client = do_create_client();

}  // namespace
