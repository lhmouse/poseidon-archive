// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_ABSTRACT_HTTP_SERVER_DECODER_HPP_
#define POSEIDON_HTTP_ABSTRACT_HTTP_SERVER_DECODER_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_HTTP_Server_Decoder
  : public ::asteria::Rcfwd<Abstract_HTTP_Server_Decoder>
  {
  private:
    linear_buffer m_queue;
    HTTP_Decoder_State m_state = { };

    uint8_t m_limits_loaded : 1;
    uint8_t m_final : 1;      // close connection after entity
    uint8_t m_chunked : 1;    // use HTTP/1.1 `chunked` transfer encoding
    uint8_t m_gzip : 1;       // use `gzip` content encoding
    uint8_t m_ws_pmce : 1;    // use WebSocket per-message compression extension
    uint8_t m_ws_nctxto : 1;  // has WebSocket `client_no_context_takeover`

    uint32_t m_max_header_length = 16384;
    uint64_t m_max_content_length = 2097152;
    uint64_t m_max_ws_data_length = 65536;

    uint64_t m_content_offset;
    uint64_t m_content_length;
    rcfwdp<zlib_Inflator> m_inflator;

  protected:
    explicit
    Abstract_HTTP_Server_Decoder() noexcept
      = default;

  private:

  protected:
    // This callback is invoked after the headers of a message have been received.
    virtual
    void
    do_http_server_on_headers(HTTP_Method meth, cow_string&& target, HTTP_Version ver,
                              Option_Map&& headers)
      = 0;

    // This callback is invoked for each chunk of the message body.
    virtual
    void
    do_http_server_on_entity(uint64_t offset, char* data, size_t size)
      = 0;

    // This callback is invoked at the end of a message.
    virtual
    void
    do_http_server_on_end_of_entity()
      = 0;

    // This callback is invoked for each chunk of data over a tunnel.
    virtual
    void
    do_http_server_on_tunnel_data(char* data, size_t size)
      = 0;

    // This callback is invoked when a tunnel has been closed.
    virtual
    void
    do_http_server_on_tunnel_closure()
      = 0;

    // This callback is invoked when a WebSocket non-closure frame has been received.
    virtual
    void
    do_http_server_on_websocket_frame(WebSocket_Opcode opcode, char* data, size_t size)
      = 0;

    // This callback is invoked when a WebSocket closure frame has been received.
    virtual
    void
    do_http_server_on_websocket_closure(WebSocket_Status stat, char* data, size_t size)
      = 0;

    // This function shall close the connection.
    virtual
    bool
    do_http_server_close()
      = 0;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_HTTP_Server_Decoder);

    // Gets the state.
    HTTP_Decoder_State
    http_decoder_state()
      const noexcept
      { return this->m_state;  }

    // Consumes all input data and invoke other callbacks.
    // This is typically called by overriders of `Abstract_Stream_Socket::
    // do_socket_on_receive()`.
    bool
    http_server_decode_stream(const char* data, size_t size);

    // Marks the closure of connection.
    bool
    http_server_decode_end_of_stream();
  };

}  // namespace poseidon

#endif
