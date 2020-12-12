// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_ABSTRACT_HTTP_SERVER_ENCODER_HPP_
#define POSEIDON_HTTP_ABSTRACT_HTTP_SERVER_ENCODER_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_HTTP_Server_Encoder
  : public ::asteria::Rcfwd<Abstract_HTTP_Server_Encoder>
  {
  private:
    HTTP_Encoder_State m_state = { };

    uint8_t m_final : 1;      // close connection after entity
    uint8_t m_chunked : 1;    // use HTTP/1.1 `chunked` transfer encoding
    uint8_t m_gzip : 1;       // use `gzip` content encoding
    uint8_t m_ws_pmce : 1;    // use WebSocket per-message compression extension
    uint8_t m_ws_nctxto : 1;  // has WebSocket `server_no_context_takeover`

    rcfwdp<zlib_Deflator> m_deflator;

  protected:
    explicit
    Abstract_HTTP_Server_Encoder()
      noexcept
      = default;

  private:
    inline
    bool
    do_encode_http_headers(HTTP_Version ver, HTTP_Status stat, const Option_Map& headers);

    inline
    bool
    do_encode_http_entity(const char* data, size_t size);

    inline
    bool
    do_finish_http_message(HTTP_Encoder_State next);

    inline
    bool
    do_encode_websocket_frame(int flags, const char* data, size_t size);

  protected:
    // This function shall deliver all bytes to the other endpoint.
    virtual
    bool
    do_http_on_server_send(const char* data, size_t size)
      = 0;

    // This function shall close the connection.
    virtual
    bool
    do_http_on_server_close()
      = 0;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_HTTP_Server_Encoder);

    // Gets the state.
    HTTP_Encoder_State
    http_encoder_state()
      const noexcept
      { return this->m_state;  }

    // Puts the HTTP status line and headers.
    // `http_encoder_state()` must be 'closed' or 'headers'.
    // `method` and `target` shall be copied from a previous request. Unless the response
    // 'MUST NOT' include a message body (see RFC 7230 section 3.3), the next state will
    // be 'entity'.
    bool
    http_encode_headers(HTTP_Version ver, HTTP_Status stat, Option_Map&& headers,
                        HTTP_Method method, const cow_string& target);

    // Puts a chunk of entity.
    // `http_encoder_state()` must be 'closed' or 'entity'.
    bool
    http_encode_entity(const char* data, size_t size);

    // Finishes the entity.
    // `http_encoder_state()` must be 'closed' or 'entity'.
    bool
    http_encode_end_of_entity();

    // Sends data through this tunnel.
    // `http_encoder_state()` must be 'closed' or 'tunnel'.
    bool
    http_encode_tunnel_data(const char* data, size_t size);

    // Shuts the entity down.
    // `http_encoder_state()` must be 'closed' or 'tunnel'.
    bool
    http_encode_tunnel_closure();

    // Puts a WebSocket frame.
    // `http_encoder_state()` must be 'closed' or 'websocket'.
    // `opcode` shall specify a valid opcode other than `websocket_opcode_continuation`
    // and `websocket_opcode_close`.
    bool
    http_encode_websocket_frame(WebSocket_Opcode opcode, const char* data, size_t size);

    // Puts a WebSocket closure frame.
    // `http_encoder_state()` must be 'closed' or 'websocket'.
    // The additional payload is appended to the status code. It is truncated to 123
    // bytes if it is longer.
    bool
    http_encode_websocket_closure(WebSocket_Status stat, const char* data, size_t size);
  };

}  // namespace poseidon

#endif
