// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../poseidon/precompiled.ipp"
#include "../poseidon/socket/ssl_socket.hpp"
#include "../poseidon/static/network_driver.hpp"
#include "../poseidon/static/async_logger.hpp"
#include "../poseidon/utils.hpp"

namespace {
using namespace poseidon;

const Socket_Address connect_address(::rocket::sref("93.184.216.34:443"));  // example.org

struct Example_Session : SSL_Socket
  {
    explicit
    Example_Session()
      : SSL_Socket(connect_address, network_driver.default_client_ssl_ctx())
      {
        this->do_ssl_alpn_request({ "http/1.1" });
        POSEIDON_LOG_FATAL(("ALPN: requesting HTTP/1.1"));
      }

    void
    do_on_ssl_connected() override
      {
        static constexpr char data[] =
            "GET / HTTP/1.1\r\n"
            "Host: example.org\r\n"
            "Connection: close\r\n"
            "\r\n";

        this->ssl_send(data, ::strlen(data));
        POSEIDON_LOG_ERROR(("example SSL client sent to `$1`:\n\n$2"), this->remote_address(), data);
      }

    void
    do_on_ssl_stream(linear_buffer& data) override
      {
        POSEIDON_LOG_FATAL(("ALPN: received `$1`"), this->alpn_protocol());

        cow_string str(data.begin(), data.end());
        data.clear();
        POSEIDON_LOG_WARN(("example SSL client received from `$1`:\n\n$2"), this->remote_address(), str);
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
