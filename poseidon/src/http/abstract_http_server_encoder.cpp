// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_http_server_encoder.hpp"
#include "option_map.hpp"
#include "enums.hpp"
#include "../core/zlib_deflator.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_HTTP_Server_Encoder::
~Abstract_HTTP_Server_Encoder()
  {
  }

bool
Abstract_HTTP_Server_Encoder::
do_encode_http_headers(HTTP_Version ver, HTTP_Status stat, const Option_Map& headers)
  {
    // Send 200 in the case of `http_status_connection_established`.
    HTTP_Status real_stat = stat;
    if(stat == http_status_connection_established)
      real_stat = http_status_ok;

    // Compose the status line followed by all headers.
    ::rocket::tinyfmt_str fmt;
    fmt << format_http_version(ver) << ' ' << real_stat << ' '
        << describe_http_status(stat) << "\r\n";

    for(const auto& pair : headers)
      fmt << pair.first << ": " << pair.second << "\r\n";
    fmt << "\r\n";

    return this->do_http_on_server_send(fmt.c_str(), fmt.length());
  }

bool
Abstract_HTTP_Server_Encoder::
do_encode_http_entity(const char* data, size_t size)
  {
    // Don't send empty chunks. Test the connection state only.
    if(size == 0)
      return this->do_http_on_server_send("", 0);

    // For HTTP/1.0, send outgoing data verbatim.
    if(!this->m_chunked)
      return this->do_http_on_server_send(data, size);

    // For HTTP/1.1, encode the data as a single chunk.
    ::rocket::ascii_numput nump;
    nump.put_XU(size);

    return this->do_http_on_server_send(nump.data() + 2, nump.size()) &&
           this->do_http_on_server_send("\r\n", 2) &&
           this->do_http_on_server_send(data, size) &&
           this->do_http_on_server_send("\r\n", 2);
  }

bool
Abstract_HTTP_Server_Encoder::
do_finish_http_message(HTTP_Encoder_State next)
  {
    bool sent = this->do_http_on_server_send("", 0);

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
      sent = sent && this->do_http_on_server_send("0\r\n\r\n", 5);

    // If this is the final message, shut the connection down.
    this->m_state = next;
    if(this->m_final) {
      this->m_state = http_encoder_state_closed;
      sent = sent && this->do_http_on_server_close();
    }
    return sent;
  }

bool
Abstract_HTTP_Server_Encoder::
do_encode_websocket_frame(int flags, const char* data, size_t size)
  {
    // Compose the frame header.
    ::rocket::static_vector<char, 14> head;
    head.emplace_back(flags | 1);  // opcode, flags, FIN

    size_t exlen;
    if(size <= 125) {
      head.emplace_back(size);  // payload length
      exlen = 0;
    }
    else if(size <= 65535) {
      head.emplace_back(126);  // payload length
      exlen = 2;
    }
    else {
      head.emplace_back(127);  // payload length
      exlen = 8;
    }
    while(exlen != 0)
      head.emplace_back(static_cast<uint64_t>(size) >> --exlen * 8);

    return this->do_http_on_server_send(head.data(), head.size()) &&
           this->do_http_on_server_send(data, size);
  }

bool
Abstract_HTTP_Server_Encoder::
http_encode_headers(HTTP_Version ver, HTTP_Status stat, Option_Map&& headers,
                    HTTP_Method method, const cow_string& target)
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_headers)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'headers')");

    this->m_final = false;
    this->m_chunked = false;
    this->m_gzip = false;

    // Check for special cases where no content shall be sent.
    // Refer to RFC 7230 section 3.3 for details.
    bool no_content = (method == http_method_connect) ||
            (classify_http_status(stat) == http_status_class_information) ||
            ::rocket::is_any_of(stat,
                { http_status_no_content, http_status_not_modified });

    if(no_content) {
      // In this case, erase all content headers.
      if(headers.erase(sref("Content-Length")))
        POSEIDON_LOG_WARN("`Content-Length` not allowed without a content");

      if(headers.erase(sref("Transfer-Encoding")))
        POSEIDON_LOG_ERROR("`Transfer-Encoding` not allowed without a content");

      if(headers.erase(sref("Content-Type")))
        POSEIDON_LOG_WARN("`Content-Type` not allowed without a content");

      if(headers.erase(sref("Content-Encoding")))
        POSEIDON_LOG_WARN("`Content-Encoding` not allowed without a content");

      if(headers.erase(sref("Content-Range")))
        POSEIDON_LOG_WARN("`Content-Range` not allowed without a content");
    }

    HTTP_Connection conn = http_connection_keep_alive;
    bool proxy = target[0] != '/';
    Option_Map opts;

    auto connection_header_name = sref("Connection");
    if(proxy)
      connection_header_name = sref("Proxy-Connection");

    // The CONNECT method is very, very special.
    if(method == http_method_connect) {
      // If a 2xx status code is sent, a tunnel will be established.
      if(classify_http_status(stat) == http_status_class_success) {
        // Erase forbidden headers.
        headers.erase(connection_header_name);
        headers.erase(sref("Content-Length"));
        headers.erase(sref("Transfer-Encoding"));

        return this->do_encode_http_headers(ver, stat, headers) &&
               this->do_finish_http_message(http_encoder_state_tunnel);
      }

      // Otherwise, the connection shall be closed immediately.
      conn = http_connection_close;
    }

    if(ver < http_version_1_1)
      conn = http_connection_close;

    if(conn == http_connection_keep_alive)
      headers.for_each(connection_header_name,
          [&](const cow_string& resph) {
            // XXX: Options may overwrite each other.
            if(stat == http_status_switching_protocol)
              if(ascii_ci_has_token(resph, sref("upgrade")))
                conn = http_connection_upgrade;

            if(ascii_ci_has_token(resph, sref("keep-alive")))
              conn = http_connection_keep_alive;

            if(ascii_ci_has_token(resph, sref("close")))
              conn = http_connection_close;
          });

    if(ver < http_version_1_1) {
      // HTTP/1.0 does not support this header.
      headers.erase(sref("Transfer-Encoding"));
    }
    else {
      // Check whether compression can be enabled.
      auto& trans_enc_str = headers.open(sref("Transfer-Encoding"));
      this->m_gzip = ascii_ci_has_token(trans_enc_str, sref("gzip"));

      // Rewrite this header.
      // The `chunked` encoding is enforced for simplicity.
      this->m_chunked = true;
      trans_enc_str = sref("chunked");

      if(this->m_gzip)
        trans_enc_str.insert(0, "gzip, ");
    }

    // Check for upgradable connections.
    // First, look at `Connection:`. Only if an `upgrade` token exists, shall we
    // check `Upgrade:` for available protocols. Note that code above might have
    // set `Connection: close`, in which case no upgrade is possible.
    if(conn == http_connection_upgrade) {
      auto upgrade_str = headers.find_opt(sref("Upgrade"));
      if(!upgrade_str)
        POSEIDON_THROW("`Connection: upgrade` sent without an `Upgrade:` header");

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
                  this->m_ws_nctxto = !!opts.count(sref("server_no_context_takeover"));
                }
              }
            });

        // WebSocket uses a raw deflate compressor, while HTTP uses GZIP.
        // It cannot be reused so delete it.
        this->m_deflator = nullptr;

        // Switch to WebSocket after the response headers.
        return this->do_encode_http_headers(ver, stat, headers) &&
               this->do_finish_http_message(http_encoder_state_websocket);
      }
      else
        POSEIDON_THROW("Protocol `$1` not upgradable", *upgrade_str);
    }

    // Rewrite default headers when the connection should be closed.
    // This must be the last operation before encoding the headers.
    if(conn == http_connection_close) {
      this->m_final = true;
      headers.erase(sref("Upgrade"));
      headers.set(connection_header_name, sref("close"));
    }

    if(no_content || (method == http_method_head))
      return this->do_encode_http_headers(ver, stat, headers) &&
             this->do_finish_http_message(http_encoder_state_headers);

    // Reset the deflator used by the previous message, if any.
    auto defl = unerase_pointer_cast<zlib_Deflator>(this->m_deflator);
    if(defl)
      defl->reset();

    // Expect the entity.
    bool sent = this->do_encode_http_headers(ver, stat, headers);
    this->m_state = http_encoder_state_entity;
    return sent;
  }

bool
Abstract_HTTP_Server_Encoder::
http_encode_entity(const char* data, size_t size)
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_entity)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'entity')");

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
Abstract_HTTP_Server_Encoder::
http_encode_end_of_entity()
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_entity)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'entity')");

    // Finish this response message and expect the next one.
    return this->do_finish_http_message(http_encoder_state_headers);
  }

bool
Abstract_HTTP_Server_Encoder::
http_encode_tunnel_data(const char* data, size_t size)
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_tunnel)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'tunnel')");

    // Forward outgoing data verbatim.
    return this->do_http_on_server_send(data, size);
  }

bool
Abstract_HTTP_Server_Encoder::
http_encode_tunnel_closure()
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_tunnel)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'tunnel')");

    // Shut the tunnel down.
    this->m_state = http_encoder_state_closed;
    return this->do_http_on_server_close();
  }

bool
Abstract_HTTP_Server_Encoder::
http_encode_websocket_frame(WebSocket_Opcode opcode, const char* data, size_t size)
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_websocket)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'websocket')");

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
      // Encode a control frame.
      // Control frames are not compressed, and must not be fragmented.
      size_t rlen = ::std::min<size_t>(size, 125);
      if(rlen != size)
        POSEIDON_LOG_WARN("Control frame truncated (size `$1`)", size);

      return this->do_encode_websocket_frame(opcode, data, rlen);
    }

    // If compression is not enabled, send outgoing data verbatim.
    if(!this->m_gzip)
      return this->do_encode_websocket_frame(opcode, data, size);

    // Compress outgoing data using deflate.
    auto defl = unerase_pointer_cast<zlib_Deflator>(this->m_deflator);
    if(!defl) {
      defl = ::rocket::make_refcnt<zlib_Deflator>(zlib_Deflator::format_raw);
      this->m_deflator = defl;
    }
    else if(this->m_ws_nctxto) {  // `server_no_context_takeover`
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

    bool sent = this->do_encode_websocket_frame(opcode | 2, obuf.data(), rlen);
    obuf.clear();
    return sent;
  }

bool
Abstract_HTTP_Server_Encoder::
http_encode_websocket_closure(WebSocket_Status stat, const char* data, size_t size)
  {
    if(this->m_state == http_encoder_state_closed)
      return false;

    if(this->m_state != http_encoder_state_websocket)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'websocket')");

    // Truncate the payload, as control frames cannot be fragmented.
    size_t rlen = ::std::min<size_t>(size, 123);
    if(rlen != size)
      POSEIDON_LOG_WARN("Closure frame truncated (size `$1`)", size);

    ::rocket::static_vector<char, 125> pbuf;
    pbuf.emplace_back(stat >> 8);  // status (high)
    pbuf.emplace_back(stat);  // status (low)
    pbuf.append(data, data + size);

    // Send the closure frame and shut the connection down.
    bool sent = this->do_encode_websocket_frame(0x80, pbuf.data(), pbuf.size());
    this->m_state = http_encoder_state_closed;
    sent = sent && this->do_http_on_server_close();
    return sent;
  }

}  // namespace poseidon
