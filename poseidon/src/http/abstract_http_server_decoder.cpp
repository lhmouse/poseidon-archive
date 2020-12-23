// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_http_server_decoder.hpp"
#include "enums.hpp"
#include "../static/main_config.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_HTTP_Server_Decoder::
~Abstract_HTTP_Server_Decoder()
  {
  }

bool
Abstract_HTTP_Server_Decoder::
http_server_decode_stream(const char* data, size_t size)
  {
    if(this->m_state == http_decoder_state_closed)
      return false;

    // Get decoder limits.
    // This operation is only performed once for each connection.
    if(!this->m_limits_loaded) {
      const auto file = Main_Config::copy();

      auto qint = file.get_int64_opt({"network","http","max_header_length"});
      if(qint)
        this->m_max_header_length = clamp_cast<uint32_t>(*qint, 256, 1048576);

      qint = file.get_int64_opt({"network","http","max_content_length"});
      if(qint)
        this->m_max_content_length = clamp_cast<uint64_t>(*qint, 0, INT64_MAX);

      qint = file.get_int64_opt({"network","http","max_websocket_data_length"});
      if(qint)
        this->m_max_ws_data_length = clamp_cast<uint64_t>(*qint, 0, INT64_MAX);

      POSEIDON_LOG_DEBUG("HTTP server decoder limits loaded successfully");
      this->m_limits_loaded = 1;
    }

    // As it is possible for multiple messages to arrive simultaneously, we have to
    // try parsing repeatedly. Each iteration shall consume some data or update the
    // decoder state. Only when no progess can be made shall the loop terminate.
    this->m_queue.putn(data, size);

    for(;;) {
      const auto state_old = this->m_state;
      const auto size_old = this->m_queue.size();
      if(size_old == 0)
        break;

      switch(this->m_state) {
        case http_decoder_state_headers:
          break;

        case http_decoder_state_closed:
          break;

        case http_decoder_state_entity:
          break;

        case http_decoder_state_tunnel:
          // Send all data through the tunnel verbatim.
          this->do_http_server_on_tunnel_data(this->m_queue.mut_begin(),
                                    this->m_queue.size());
          this->m_queue.clear();
          break;

        case http_decoder_state_websocket:
          break;
      }

      if(this->m_state == http_decoder_state_closed)
        break;

      if((this->m_state == state_old) && (this->m_queue.size() == size_old))
        break;
    }
    return true;
  }

bool
Abstract_HTTP_Server_Decoder::
http_server_decode_end_of_stream()
  {
    if(this->m_state == http_decoder_state_closed)
      return false;

    switch(this->m_state) {
      case http_decoder_state_headers:
      case http_decoder_state_closed:
      case http_decoder_state_entity:
        break;

      case http_decoder_state_tunnel:
        // Notify closure of the tunnel.
        this->do_http_server_on_tunnel_closure();
        break;

      case http_decoder_state_websocket:
        // Notify closure of the WebSocket connection. Here the connection
        // has been closed but no WebSocket closure frame was received.
        this->do_http_server_on_websocket_closure(
                   websocket_status_no_status, reinterpret_cast<char*>(-1), 0);
        break;
    }

    // Discard pending data and shut the connection down.
    this->m_state = http_decoder_state_closed;
    this->do_http_server_close();
    return true;
  }

}  // namespace poseidon
