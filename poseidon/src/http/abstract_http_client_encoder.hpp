// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_ABSTRACT_HTTP_CLIENT_ENCODER_HPP_
#define POSEIDON_HTTP_ABSTRACT_HTTP_CLIENT_ENCODER_HPP_

#include "../fwd.hpp"
#include "../core/lcg48.hpp"

namespace poseidon {

class Abstract_HTTP_Client_Encoder
  : public ::asteria::Rcfwd<Abstract_HTTP_Client_Encoder>
  {
  private:
    HTTP_Encoder_State m_state = { };
    bool m_good = true;

    uint8_t m_final : 1;      // close/upgrade connection after entity
    uint8_t m_upgrading : 1;  // upgrade connection after entity
    uint8_t m_chunked : 1;    // use HTTP/1.1 chunked encoding
    uint8_t m_gzip : 1;       // use GZIP transfer encoding
    uint8_t m_ws_pmce : 1;    // use WebSocket per-message compression extension
    uint8_t m_ws_nctxto : 1;  // has WebSocket `client_no_context_takeover`

    struct Pipelined_Request
      {
        HTTP_Method meth;
        HTTP_Connection conn;
        uint8_t proxy : 1;
      };

    ::rocket::cow_vector<Pipelined_Request> m_pipeline;
    LCG48 m_random;
    rcfwdp<zlib_Deflator> m_deflator;

  public:
    explicit
    Abstract_HTTP_Client_Encoder()
      noexcept
      = default;

  private:
    inline
    void
    do_encode_http_headers(HTTP_Method meth, const cow_string& target, HTTP_Version ver,
                           const Option_Map& headers, HTTP_Connection conn);

    inline
    void
    do_encode_http_entity(const char* data, size_t size);

    inline
    void
    do_finish_http_message();

    inline
    void
    do_encode_websocket_frame(int flags, char* data, size_t size);  // data clobbered

  protected:
    // This function shall deliver all bytes to the other endpoint.
    virtual
    bool
    do_http_client_send(const char* data, size_t size)
      = 0;

    // This function shall close the connection.
    virtual
    bool
    do_http_client_close()
      = 0;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_HTTP_Client_Encoder);

    // Gets the state.
    HTTP_Encoder_State
    http_encoder_state()
      const noexcept
      { return this->m_state;  }

    // Puts the HTTP request line and headers.
    // `http_encoder_state()` must be 'closed' or 'headers'.
    // Unless the request includes a message body (see RFC 7230 section 3.3), the next
    // state will be 'entity'.
    bool
    http_encode_headers(HTTP_Method meth, const cow_string& target, HTTP_Version ver,
                        Option_Map&& headers);

    // Puts a chunk of entity. Note that it is not valid to switch to a new protocol
    // without receipt of a successful response.
    // `http_encoder_state()` must be 'closed' or 'entity'.
    bool
    http_encode_entity(const char* data, size_t size);

    // Finishes the entity. Note that it is not valid to switch to a new protocol
    // without receipt of a successful response.
    // `http_encoder_state()` must be 'closed' or 'entity'.
    bool
    http_encode_end_of_entity();

    // Sends data through this tunnel.
    // `http_encoder_state()` must be 'closed' or 'tunnel'.
    bool
    http_encode_tunnel_data(const char* data, size_t size);

    // Shuts the tunnel down.
    // `http_encoder_state()` must be 'closed' or 'tunnel'.
    bool
    http_encode_tunnel_closure();

    // Accepts the header of a response message.
    // WARNING: This function may close the connection arbitrarily.
    bool
    http_on_response_headers(HTTP_Status stat, const Option_Map& headers);

    // Puts a WebSocket frame.
    // `http_encoder_state()` must be 'closed' or 'websocket'.
    // Note that it is not valid to switch to a new protocol without receipt of a
    // successful response.
    bool
    http_encode_websocket_frame(WebSocket_Opcode opcode, const char* data, size_t size);

    // Puts a WebSocket closure frame.
    // `http_encoder_state()` must be 'closed' or 'websocket'.
    // Note that it is not valid to switch to a new protocol without receipt of a
    // successful response.
    bool
    http_encode_websocket_closure(WebSocket_Status stat, const char* data, size_t size);
  };

}  // namespace poseidon

#endif
