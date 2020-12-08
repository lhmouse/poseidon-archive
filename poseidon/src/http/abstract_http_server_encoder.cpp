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

    return this->do_http_on_server_send(nump.data(), nump.size()) &&
           this->do_http_on_server_send("\r\n", 2) &&
           this->do_http_on_server_send(data, size) &&
           this->do_http_on_server_send("\r\n", 2);
  }

bool
Abstract_HTTP_Server_Encoder::
do_finish_http_message(Encoder_State next)
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
      this->m_state = encoder_state_closed;
      sent = sent && this->do_http_on_server_close();
    }
    return sent;
  }

bool
Abstract_HTTP_Server_Encoder::
do_encode_websocket_frame(uint8_t flags, WebSocket_Opcode opcode,
                          const char* data, size_t size)
  {
    // Compose the frame header.
    ::rocket::static_vector<char, 14> head;
    head.emplace_back(opcode << 4 | flags | 1);  // opcode, flags, FIN

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
http_encode_headers(HTTP_Status stat, Option_Map&& headers, HTTP_Method req_method,
                    const cow_string& req_target, HTTP_Version req_ver,
                    const Option_Map& req_headers)
  {
    if(this->m_state == encoder_state_closed)
      return false;

    if(this->m_state != encoder_state_headers)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'headers')");

    // Get the HTTP version that the response will use.
    // This will be either HTTP/1.0 or HTTP/1.1.
    HTTP_Version ver = ::rocket::clamp(req_ver, http_version_1_0, http_version_1_1);

    // Check for special cases where no content shall be sent.
    // Refer to RFC 7230 section 3.3 for details.
    bool no_content = (req_method == http_method_connect) ||
            (classify_http_status(stat) == http_status_class_information) ||
            ::rocket::is_any_of(stat,
                { http_status_no_content, http_status_not_modified });

    this->m_final = false;
    this->m_chunked = false;
    this->m_gzip = false;

    // In this case, erase all content headers.
    if(no_content) {
      if(headers.erase(sref("Content-Type")))
        POSEIDON_LOG_WARN("`Content-Type` not allowed without a content");

      if(headers.erase(sref("Content-Length")))
        POSEIDON_LOG_WARN("`Content-Length` not allowed without a content");

      if(headers.erase(sref("Content-Encoding")))
        POSEIDON_LOG_WARN("`Content-Encoding` not allowed without a content");

      if(headers.erase(sref("Content-Range")))
        POSEIDON_LOG_WARN("`Content-Range` not allowed without a content");
    }

    // Erase hop-to-hop headers.
    if(headers.erase(sref("Transfer-Encoding")))
      POSEIDON_LOG_ERROR("`Transfer-Encoding` not customizable");

    // CONNECT is used to establish a tunnel. The connection must not be closed.
    if(req_method == http_method_connect) {
      // If a 2xx status code is sent, a tunnel will be established.
      this->m_final = classify_http_status(stat) != http_status_class_success;
      if(this->m_final)
        headers.set(sref("Connection"), sref("close"));

      return this->do_encode_http_headers(ver, stat, headers) &&
             this->do_finish_http_message(encoder_state_tunnel);
    }

    HTTP_Upgrade upg = http_upgrade_null;
    Option_Map opts;
    ::rocket::ascii_numget numg;

    auto connection_header_name = sref("Connection");
    if(req_target[0] != '/')
      connection_header_name = sref("Proxy-Connection");

    if(ver < http_version_1_1) {
      // HTTP/1.0 does not support persistent connections.
      this->m_final = true;
      headers.set(connection_header_name, sref("close"));
    }
    else {
      // If a closure header is sent, shut the connection down.
      // For HTTP/1.1, persistent connections are enabled by default.
      this->m_final = !!req_headers.find_if_opt(connection_header_name,
          [&](const cow_string& reqh) {
            return ascii_ci_has_token(reqh, sref("close"));
          });
      if(this->m_final)
        headers.set(connection_header_name, sref("close"));

      if(!no_content) {
        // The `chunked` encoding is enforced for simplicity.
        this->m_chunked = true;
        headers.set(sref("Transfer-Encoding"), sref("chunked"));
      }

      // Check for upgradable connections.
      // First, look at `Connection:`. Only if an `upgrade` token exists, shall we
      // check `Upgrade:` for available protocols. Note that code above might have
      // set `Connection: close`, in which case no upgrade is possible.
      auto upgrade_str = headers.find_if_opt(connection_header_name,
                [&](const cow_string& resph) {
                  return ascii_ci_has_token(resph, sref("upgrade"));
                });
      if(upgrade_str)
        upgrade_str = headers.find_opt(sref("Upgrade"));

      if(upgrade_str) {
        // This upgrade mechanism is utilized by WebSocket and HTTP/2.
        // Note we don't check whether this was actually requested by the client,
        // because we are unable to fabricate a failure response if it is proved to
        // be the case. It is the caller that has to ensure the response is valid.
        // TODO: At the moment, only WebSocket is supported.
        if(ascii_ci_equal(*upgrade_str, sref("websocket"))) {
          // Check whether compression can be enabled.
          // Refer to RFC 7692 for details.
          this->m_ws_pmce = !!headers.find_if_opt(sref("Sec-WebSocket-Extensions"),
              [&](const cow_string& resph) {
                size_t comma = 0;
                while(comma != resph.size()) {
                  opts.parse_http_header(&comma, resph, 1);
                  auto qstr = opts.find_opt(sref(""));  // extension name
                  if(!qstr)
                    continue;

                  if(ascii_ci_equal(*qstr, sref("permessage-deflate"))) {
                    // Check for context options.
                    // FIXME: RFC 7692 says if an option is not supported, the
                    //        extension must not be used. But we just accept and
                    //        ignore them here.
                    qstr = opts.find_opt(sref("server_no_context_takeover"));
                    this->m_ws_nctxto = !!qstr;
                    return true;
                  }
                }
                return false;
              });

          // Upgrade to WebSocket.
          upg = http_upgrade_websocket;
        }
        else
          POSEIDON_LOG_ERROR("Protocol `$1` not upgradable", *upgrade_str);

        // If upgrade is not possible, fail.
        if(upg == http_upgrade_null) {
          this->m_final = true;
          headers.set(connection_header_name, sref("close"));

          upgrade_str = nullptr;
          headers.erase(sref("Upgrade"));
        }
      }
    }

    // Check whether the entity can be compressed.
    // This is only examined if no explicit `Content-Encoding` has been set.
    if(!no_content && !headers.find_opt(sref("Content-Encoding"))) {
      // At the moment only `gzip` is supported.
      this->m_gzip = !!req_headers.find_if_opt(sref("Accept-Encoding"),
          [&](const cow_string& reqh) {
            size_t comma = 0;
            while(comma != reqh.size()) {
              opts.parse_http_header(&comma, reqh, 1);
              auto qstr = opts.find_opt(sref(""));  // algorithm
              if(!qstr)
                continue;

              if(ascii_ci_equal(*qstr, sref("gzip"))) {  // only 'gzip'
                // GZIP is enabled unless suppressed by a zero quality value.
                double q = 1.0;
                qstr = opts.find_opt(sref("q"));
                if(qstr) {
                  const char* sp = qstr->data();
                  if(numg.parse_F(sp, qstr->data() + qstr->size(), 10))
                    numg.cast_F(q, 0.0, 1.0);
                }
                if(q > 0)
                  return true;
              }
            }
            return false;
          });
      if(this->m_gzip)
        headers.set(sref("Content-Encoding"), sref("gzip"));
    }

    // Encode response headers now.
    if(no_content || (req_method == http_method_head))
      return this->do_encode_http_headers(ver, stat, headers) &&
             this->do_finish_http_message(encoder_state_headers);

    // Check for upgradable connections.
    if(upg == http_upgrade_websocket) {
      // WebSocket uses a raw deflate compressor, while HTTP uses GZIP.
      // It cannot be reused so delete it.
      this->m_deflator = nullptr;

      // Switch to WebSocket after the response headers.
      return this->do_encode_http_headers(ver, stat, headers) &&
             this->do_finish_http_message(encoder_state_websocket);
    }

    // Reset the deflator used by the previous message, if any.
    auto defl = unerase_pointer_cast<zlib_Deflator>(this->m_deflator);
    if(defl)
      defl->reset();

    // Expect the entity.
    bool sent = this->do_encode_http_headers(ver, stat, headers);
    this->m_state = encoder_state_entity;
    return sent;
  }

bool
Abstract_HTTP_Server_Encoder::
http_encode_entity(const char* data, size_t size)
  {
    if(this->m_state == encoder_state_closed)
      return false;

    if(this->m_state != encoder_state_entity)
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
    if(this->m_state == encoder_state_closed)
      return false;

    if(this->m_state != encoder_state_entity)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'entity')");

    // Finish this response message and expect the next one.
    return this->do_finish_http_message(encoder_state_headers);
  }

bool
Abstract_HTTP_Server_Encoder::
http_encode_tunnel_data(const char* data, size_t size)
  {
    if(this->m_state == encoder_state_closed)
      return false;

    if(this->m_state != encoder_state_tunnel)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'tunnel')");

    // Forward outgoing data verbatim.
    return this->do_http_on_server_send(data, size);
  }

bool
Abstract_HTTP_Server_Encoder::
http_encode_tunnel_closure()
  {
    if(this->m_state == encoder_state_closed)
      return false;

    if(this->m_state != encoder_state_tunnel)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'tunnel')");

    // Shut the tunnel down.
    this->m_state = encoder_state_closed;
    return this->do_http_on_server_close();
  }

bool
Abstract_HTTP_Server_Encoder::
http_encode_websocket_frame(WebSocket_Opcode opcode, const char* data, size_t size)
  {
    if(this->m_state == encoder_state_closed)
      return false;

    if(this->m_state != encoder_state_websocket)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'websocket')");

    if(opcode == websocket_opcode_close) {
      // Send a default status code of 1000 and shut down the connection.
      // This use is discouraged.
      POSEIDON_LOG_ERROR("Use `http_encode_websocket_closure()` to shut down "
                         "WebSocket connections");

      return this->http_encode_websocket_closure(
                       websocket_status_normal_closure, data, size);
    }

    if(classify_websocket_opcode(opcode) == websocket_opcode_class_control) {
      // Truncate the payload, as control frames cannot be fragmented.
      size_t rlen = ::std::min<size_t>(size, 125);
      if(rlen != size)
        POSEIDON_LOG_WARN("Control frame truncated (size `$1`)", size);

      return this->do_encode_websocket_frame(0, opcode, data, rlen);
    }

    // If compression is not enabled, send outgoing data verbatim.
    if(!this->m_gzip)
      return this->do_encode_websocket_frame(0, opcode, data, size);

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

    bool sent = this->do_encode_websocket_frame(2, opcode, obuf.data(), rlen);
    obuf.clear();
    return sent;
  }

bool
Abstract_HTTP_Server_Encoder::
http_encode_websocket_closure(WebSocket_Status stat, const char* data, size_t size)
  {
    if(this->m_state == encoder_state_closed)
      return false;

    if(this->m_state != encoder_state_websocket)
      POSEIDON_THROW("HTTP server encoder state error (expecting 'websocket')");

    // Truncate the payload, as control frames cannot be fragmented.
    size_t rlen = ::std::min<size_t>(size, 123);
    if(rlen != size)
      POSEIDON_LOG_WARN("Closure frame truncated (size `$1`)", size);

    ::rocket::static_vector<char, 125> payload;
    payload.emplace_back(stat >> 8);  // status (high)
    payload.emplace_back(stat);  // status (low)
    payload.append(data, data + size);

    // Send the closure frame and shut the connection down.
    bool sent = this->do_encode_websocket_frame(0, websocket_opcode_close,
                                                payload.data(), payload.size());
    this->m_state = encoder_state_closed;
    sent = sent && this->do_http_on_server_close();
    return sent;
  }

}  // namespace poseidon
