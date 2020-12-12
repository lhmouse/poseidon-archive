// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_http_client_encoder.hpp"
#include "option_map.hpp"
#include "enums.hpp"
#include "../core/zlib_deflator.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_HTTP_Client_Encoder::
~Abstract_HTTP_Client_Encoder()
  {
  }

bool
Abstract_HTTP_Client_Encoder::
do_encode_http_headers(HTTP_Method method, const cow_string& target, HTTP_Version ver,
                       const Option_Map& headers, HTTP_Connection conn)
  {
    // Pipeline this request.
    Pipelined_Request req = { method, conn, target[0] != '/' };
    this->m_pipeline.emplace_back(req);

    // Compose the request line followed by all headers.
    // Note we don't check whether `target` is a valid URI or not.
    ::rocket::tinyfmt_str fmt;
    fmt << format_http_method(method) << ' ' << target << ' '
        << format_http_version(ver) << "\r\n";

    for(const auto& pair : headers)
      fmt << pair.first << ": " << pair.second << "\r\n";
    fmt << "\r\n";

    return this->do_http_on_client_send(fmt.c_str(), fmt.length());
  }

bool
Abstract_HTTP_Client_Encoder::
do_encode_http_entity(const char* data, size_t size)
  {
    // Don't send empty chunks. Test the connection state only.
    if(size == 0)
      return this->do_http_on_client_send("", 0);

    // For HTTP/1.0, send outgoing data verbatim.
    if(!this->m_chunked)
      return this->do_http_on_client_send(data, size);

    // For HTTP/1.1, encode the data as a single chunk.
    ::rocket::ascii_numput nump;
    nump.put_XU(size);

    return this->do_http_on_client_send(nump.data(), nump.size()) &&
           this->do_http_on_client_send("\r\n", 2) &&
           this->do_http_on_client_send(data, size) &&
           this->do_http_on_client_send("\r\n", 2);
  }

bool
Abstract_HTTP_Client_Encoder::
do_finish_http_message(HTTP_Encoder_State next)
  {
    bool sent = this->do_http_on_client_send("", 0);

    // Terminate the entity.
    if(this->m_gzip) {
      auto defl = unerase_pointer_cast<zlib_Deflator>(this->m_deflator);
      if(defl) {
        defl->finish();
        auto& obuf = defl->output_buffer();
        sent = sent && this->do_encode_http_entity(obuf.data(), obuf.size());
        obuf.clear();
      }
    }

    if(this->m_chunked)
      sent = sent && this->do_http_on_client_send("0\r\n\r\n", 5);

    // If this is the final message, mark the connection for closure.
    // Don't call `do_http_on_client_close()` because clients should not close
    // connections prematurely. This connection will be closed after the its
    // response is acknowledged in `Abstract_HTTP_Client_Decoder`.
    this->m_state = next;
    if(this->m_final) {
      this->m_state = this->m_upgrading ? http_encoder_state_upgrading
                                        : http_encoder_state_closed;
    }
    return sent;
  }

bool
Abstract_HTTP_Client_Encoder::
do_encode_websocket_frame(int flags, char* data, size_t size)
  {
    // Compose the frame header.
    ::rocket::static_vector<char, 14> head;
    head.emplace_back(flags | 1);  // opcode, flags, FIN

    size_t exlen;
    if(size <= 125) {
      head.emplace_back(128 + size);  // MASK, payload length
      exlen = 0;
    }
    else if(size <= 65535) {
      head.emplace_back(254);  // MASK, payload length
      exlen = 2;
    }
    else {
      head.emplace_back(255);  // MASK, payload length
      exlen = 8;
    }
    while(exlen != 0)
      head.emplace_back(static_cast<uint64_t>(size) >> --exlen * 8);

    // Create a 4-byte mask and mask the payload.
    // Client-to-server messages must be masked according to RFC 6455.
    uint32_t mask = this->m_random.bump();
    exlen = 4;
    while(exlen != 0)
      head.emplace_back(mask >> --exlen * 8);  // mask in big-endian order

    exlen = 0;
    while(exlen != size)
      mask = mask << 8 | mask >> 24,
        data[exlen++] ^= static_cast<char>(mask);

    return this->do_http_on_client_send(head.data(), head.size()) &&
           this->do_http_on_client_send(data, size);
  }

bool
Abstract_HTTP_Client_Encoder::
http_encode_headers(HTTP_Method method, const cow_string& target, HTTP_Version ver,
                    Option_Map&& headers)
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_headers)
      POSEIDON_THROW("HTTP client encoder state error (expecting 'headers')");

    this->m_final = false;
    this->m_upgrading = false;
    this->m_chunked = false;
    this->m_gzip = false;

    HTTP_Connection conn = http_connection_keep_alive;
    bool proxy = target[0] != '/';
    Option_Map opts;

    auto connection_header_name = sref("Connection");
    if(proxy)
      connection_header_name = sref("Proxy-Connection");

    // The CONNECT method is very, very special.
    if(method == http_method_connect) {
      // Erase forbidden headers.
      // This request should have no content.
      headers.erase(connection_header_name);
      headers.erase(sref("Content-Length"));
      headers.erase(sref("Transfer-Encoding"));

      return this->do_encode_http_headers(method, target, ver, headers, conn) &&
             this->do_finish_http_message(http_encoder_state_upgrading);
    }

    // Look for `Transfer-Encoding:` or `Content-Length:`.
    // If both are specified, the former takes precedence.
    bool no_content = false;
    auto mut_qstr = headers.mut_find_opt(sref("Transfer-Encoding"));
    if(mut_qstr) {
      // Note that there is no way in HTTP/1.0 to send a request message with an
      // indeterminate number of bytes. The server encoder in this case deletes
      // `Transfer-Encoding:` and works in the traditional one-message-per-connection
      // mode, which is unfortunately invalid on the client side.
      if(ver < http_version_1_1)
        POSEIDON_THROW("`Transfer-Encoding` cannot be specified in HTTP/1.0");

      // Check whether compression can be enabled.
      auto& trans_enc_str = *mut_qstr;
      this->m_gzip = ascii_ci_has_token(trans_enc_str, sref("gzip"));

      // Rewrite this header.
      this->m_chunked = true;
      trans_enc_str = sref("chunked");

      if(this->m_gzip)
        trans_enc_str.insert(0, "gzip, ");
    }
    else {
      // Note if `Content-Length` is not specified, it is considered to have a value
      // of zero, which means there is no content.
      uint64_t length = 0;
      auto qstr = headers.find_opt(sref("Content-Length"));
      if(qstr) {
        // Parse the value as a signed 64-bit integer.
        ::rocket::ascii_numget numg;
        const char* sp = qstr->data();
        if(!numg.parse_U(sp, sp + qstr->size(), 10))
          POSEIDON_THROW("Invalid `Content-Length` value: $1", *qstr);

        if(!numg.cast_U(length, 0, INT64_MAX))
          POSEIDON_THROW("`Content-Length` value out of range: $1", *qstr);
      }
      no_content = !length;
    }

    if(no_content) {
      // In this case, erase all content headers.
      if(headers.erase(sref("Content-Type")))
        POSEIDON_LOG_WARN("`Content-Type` not allowed without a content");

      if(headers.erase(sref("Content-Encoding")))
        POSEIDON_LOG_WARN("`Content-Encoding` not allowed without a content");

      if(headers.erase(sref("Content-Range")))
        POSEIDON_LOG_WARN("`Content-Range` not allowed without a content");
    }

    if(ver < http_version_1_1)
      conn = http_connection_close;

    if(conn == http_connection_keep_alive)
      headers.for_each(connection_header_name,
          [&](const cow_string& resph) {
            // XXX: Options may overwrite each other.
            if(ascii_ci_has_token(resph, sref("upgrade")))
              conn = http_connection_upgrade;

            if(ascii_ci_has_token(resph, sref("keep-alive")))
              conn = http_connection_keep_alive;

            if(ascii_ci_has_token(resph, sref("close")))
              conn = http_connection_close;
          });

    // Check for upgradable connections.
    // Validation and state transition are done in `http_on_response_headers()`.
    if(conn == http_connection_upgrade) {
      this->m_final = true;
      this->m_upgrading = true;
    }

    // Rewrite default headers when the connection should be closed.
    // This must be the last operation before encoding the headers.
    if(conn == http_connection_close) {
      this->m_final = true;
      headers.erase(sref("Upgrade"));
      headers.set(connection_header_name, sref("close"));
    }

    if(no_content)
      return this->do_encode_http_headers(method, target, ver, headers, conn) &&
             this->do_finish_http_message(http_encoder_state_headers);

    // Reset the deflator used by the previous message, if any.
    auto defl = unerase_pointer_cast<zlib_Deflator>(this->m_deflator);
    if(defl)
      defl->reset();

    // Expect the entity.
    bool sent = this->do_encode_http_headers(method, target, ver, headers, conn);
    this->m_state = http_encoder_state_entity;
    return sent;
  }

bool
Abstract_HTTP_Client_Encoder::
http_encode_entity(const char* data, size_t size)
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_entity)
      POSEIDON_THROW("HTTP client encoder state error (expecting 'entity')");

    // If compression is not enabled, send outgoing data verbatim.
    if(!this->m_gzip)
      return this->do_encode_http_entity(data, size);

    // Compress outgoing data using GZIP.
    auto defl = unerase_pointer_cast<zlib_Deflator>(this->m_deflator);
    if(!defl) {
      defl = ::rocket::make_refcnt<zlib_Deflator>(zlib_Deflator::format_gzip);
      this->m_deflator = defl;
    }
    defl->write(data, size);

    // Consume all compressed data.
    auto& obuf = defl->output_buffer();
    bool sent = this->do_encode_http_entity(obuf.data(), obuf.size());
    obuf.clear();
    return sent;
  }

bool
Abstract_HTTP_Client_Encoder::
http_encode_end_of_entity()
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_entity)
      POSEIDON_THROW("HTTP client encoder state error (expecting 'entity')");

    // Finish this response message and expect the next one.
    return this->do_finish_http_message(http_encoder_state_headers);
  }

bool
Abstract_HTTP_Client_Encoder::
http_encode_tunnel_data(const char* data, size_t size)
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_tunnel)
      POSEIDON_THROW("HTTP client encoder state error (expecting 'tunnel')");

    // Forward outgoing data verbatim.
    return this->do_http_on_client_send(data, size);
  }

bool
Abstract_HTTP_Client_Encoder::
http_encode_tunnel_closure()
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_tunnel)
      POSEIDON_THROW("HTTP client encoder state error (expecting 'tunnel')");

    // Shut the tunnel down.
    this->m_state = http_encoder_state_closed;
    return this->do_http_on_client_close();
  }

bool
Abstract_HTTP_Client_Encoder::
http_on_response_headers(HTTP_Status stat, const Option_Map& headers)
  {
    if(this->m_pipeline.empty())
      POSEIDON_THROW("HTTP response received without a matching request");

    if(this->m_state == http_encoder_state_closed)
      return false;

    // Get the earliest request, but don't discard it yet.
    HTTP_Method method = this->m_pipeline.front().method;
    HTTP_Connection conn = this->m_pipeline.front().conn;
    bool proxy = this->m_pipeline.front().proxy;
    Option_Map opts;

    // Check for upgradable connections.
    if(stat == http_status_switching_protocol) {
      auto upgrade_str = headers.find_opt(sref("Upgrade"));
      if(!upgrade_str)
        POSEIDON_THROW("HTTP status 101 received without an `Upgrade:` header");

      // Data that followed an upgrade request might have been sent to the target host,
      // so we abandon the connection in this case.
      if(this->m_pipeline.size() > 1)
        POSEIDON_THROW("No data shall follow an upgrade request");

      if(this->m_state != http_encoder_state_upgrading)
        POSEIDON_THROW("No data shall follow an upgrade request");

      if(ascii_ci_equal(*upgrade_str, sref("websocket"))) {
        // Check whether compression can be enabled. Refer to RFC 7692 for details.
        this->m_ws_pmce = false;
        this->m_ws_nctxto = false;

        headers.for_each(sref("Sec-WebSocket-Extensions"),
            [&](const cow_string& resph) {
              size_t comma = 0;
              while(comma != resph.size()) {
                opts.parse_http_header(&comma, resph, 1);
                auto qstr = opts.find_opt(sref(""));  // extension name
                if(!qstr)
                  continue;

                if(ascii_ci_equal(*qstr, sref("permessage-deflate"))) {
                  // Check for context options.
                  // FIXME: RFC 7692 says if an option is not supported, the extension
                  //        must not be used. But we just accept and ignore them here.
                  this->m_ws_pmce = true;
                  this->m_ws_nctxto = !!opts.count(sref("client_no_context_takeover"));
                }
              }
            });

        // WebSocket uses a raw deflate compressor, while HTTP uses GZIP.
        // It cannot be reused so delete it.
        this->m_deflator = nullptr;

        // Activate the WebSocket connection.
        this->m_state = http_encoder_state_websocket;
        this->m_pipeline.clear();
        this->m_pipeline.shrink_to_fit();
        return true;
      }
      else
        POSEIDON_THROW("Protocol `$1` not upgradable", *upgrade_str);
    }

    // Other 1xx status codes are accepted and ignored.
    if(classify_http_status(stat) == http_status_class_information)
      return true;

    // The other status codes denote final responses so pop the first pipelined request.
    this->m_pipeline.erase(this->m_pipeline.begin());

    // Check for persistent connections.
    if(conn != http_connection_close) {
      // This is done only if no `Connection: close` was requested. This connection is
      // closed othewise, regardless of what the server has returned.
      if(method == http_method_connect) {
        // Data that followed a CONNECT request might have been sent to the target host,
        // so we abandon the connection in this case.
        if(this->m_pipeline.size() > 0)
          POSEIDON_THROW("No data shall follow a CONNECT request");

        if(this->m_state != http_encoder_state_upgrading)
          POSEIDON_THROW("No data shall follow a CONNECT request");

        // If a 2xx status code has been received, a tunnel will have been established.
        // Otherwise, the connection shall be closed immediately.
        if(classify_http_status(stat) != http_status_class_success)
          POSEIDON_THROW("HTTP CONNECT failure (status $1 received)", stat);

        // Activate the tunnel.
        this->m_state = http_encoder_state_tunnel;
        this->m_pipeline.clear();
        this->m_pipeline.shrink_to_fit();
        return true;
      }

      auto connection_header_name = sref("Connection");
      if(proxy)
        connection_header_name = sref("Proxy-Connection");

      headers.for_each(connection_header_name,
          [&](const cow_string& resph) {
            // XXX: Options may overwrite each other.
            if(ascii_ci_has_token(resph, sref("upgrade")))
              conn = http_connection_close;  // not upgradable here

            if(ascii_ci_has_token(resph, sref("keep-alive")))
              conn = http_connection_keep_alive;

            if(ascii_ci_has_token(resph, sref("close")))
              conn = http_connection_close;
          });

      if(conn == http_connection_keep_alive)
        return true;
    }

    // Close the connection if it has been marked for closure by the server.
    this->m_state = http_encoder_state_closed;
    this->do_http_on_client_close();
    return true;
  }

bool
Abstract_HTTP_Client_Encoder::
http_encode_websocket_frame(WebSocket_Opcode opcode, const char* data, size_t size)
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_websocket)
      POSEIDON_THROW("HTTP client encoder state error (expecting 'websocket')");

    if(opcode == websocket_opcode_continuation)
      POSEIDON_THROW("WebSocket continuation frames cannot be sent explicitly");

    if(opcode == websocket_opcode_close) {
      // Send a default status code of 1000 and shut down the connection.
      // This use is discouraged.
      POSEIDON_LOG_ERROR("Use `http_encode_websocket_closure()` to shut down "
                         "WebSocket connections");

      return this->http_encode_websocket_closure(
                       websocket_status_normal_closure, data, size);
    }

    if(opcode >= websocket_opcode_close) {
      // Truncate the payload, as control frames cannot be fragmented.
      size_t rlen = ::std::min<size_t>(size, 125);
      if(rlen != size)
        POSEIDON_LOG_WARN("Control frame truncated (size `$1`)", size);

      return this->do_encode_websocket_frame(opcode,
           ::rocket::static_vector<char, 125>(data, data + rlen).mut_data(), rlen);
    }

    // If compression is not enabled, send outgoing data verbatim.
    if(!this->m_gzip)
      return this->do_encode_websocket_frame(opcode,
               ::rocket::cow_vector<char>(data, data + size).mut_data(), size);

    // Compress outgoing data using deflate.
    auto defl = unerase_pointer_cast<zlib_Deflator>(this->m_deflator);
    if(!defl) {
      defl = ::rocket::make_refcnt<zlib_Deflator>(zlib_Deflator::format_raw);
      this->m_deflator = defl;
    }
    else if(this->m_ws_nctxto) {  // `client_no_context_takeover`
      defl->reset();
    }
    defl->write(data, size);

    // Finish this deflate block. This results in four extra bytes `00 00 FF FF` in
    // the stream, which shall be removed according to RFC 7692. All the other bytes
    // are consumed as usual.
    defl->flush();
    auto& obuf = defl->output_buffer();

    ROCKET_ASSERT(obuf.size() >= 4);
    size_t rlen = obuf.size() - 4;
    ROCKET_ASSERT(::memcmp(obuf.data() + rlen, "\x00\x00\xFF\xFF", 4) == 0);

    bool sent = this->do_encode_websocket_frame(opcode | 2, obuf.mut_data(), rlen);
    obuf.clear();
    return sent;
  }

bool
Abstract_HTTP_Client_Encoder::
http_encode_websocket_closure(WebSocket_Status stat, const char* data, size_t size)
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_websocket)
      POSEIDON_THROW("HTTP client encoder state error (expecting 'websocket')");

    // Truncate the payload, as control frames cannot be fragmented.
    size_t rlen = ::std::min<size_t>(size, 123);
    if(rlen != size)
      POSEIDON_LOG_WARN("Closure frame truncated (size `$1`)", size);

    ::rocket::static_vector<char, 125> pbuf;
    pbuf.emplace_back(stat >> 8);  // status (high)
    pbuf.emplace_back(stat);  // status (low)
    pbuf.append(data, data + size);

    // Send the closure frame and shut the connection down.
    bool sent = this->do_encode_websocket_frame(0x80, pbuf.mut_data(), pbuf.size());
    this->m_state = http_encoder_state_closed;
    sent = sent && this->do_http_on_client_close();
    return sent;
  }

}  // namespace poseidon
