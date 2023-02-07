// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../poseidon/precompiled.ipp"
#include "../poseidon/socket/tcp_socket.hpp"
#include "../poseidon/static/network_driver.hpp"
#include "../poseidon/static/async_logger.hpp"
#include "../poseidon/utils.hpp"

namespace {
using namespace poseidon;

const Socket_Address connect_address(::rocket::sref("93.184.216.34:80"));  // example.org

struct Example_Session : TCP_Socket
  {
    explicit
    Example_Session()
      : TCP_Socket(connect_address)
      {
      }

    void
    do_on_tcp_connected() override
      {
        static constexpr char data[] =
            "GET / HTTP/1.1\r\n"
            "Host: example.org\r\n"
            "Connection: close\r\n"
            "\r\n";

        this->tcp_send(data, ::strlen(data));
        POSEIDON_LOG_ERROR(("example TCP client sent to `$1`:\n\n$2"), this->remote_address(), data);
      }

    void
    do_on_tcp_stream(linear_buffer& data) override
      {
        cow_string str(data.begin(), data.end());
        data.clear();
        POSEIDON_LOG_WARN(("example TCP client received from `$1`:\n\n$2"), this->remote_address(), str);
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
