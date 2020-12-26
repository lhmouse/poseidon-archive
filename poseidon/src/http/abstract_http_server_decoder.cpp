// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "abstract_http_server_decoder.hpp"
#include "http_exception.hpp"
#include "option_map.hpp"
#include "enums.hpp"
#include "../static/main_config.hpp"
#include "../core/config_file.hpp"
#include "../core/zlib_inflator.hpp"
#include "../utils.hpp"

namespace poseidon {

Abstract_HTTP_Server_Decoder::
~Abstract_HTTP_Server_Decoder()
  {
  }

void
Abstract_HTTP_Server_Decoder::
do_decode_http_headers()
  {
    POSEIDON_LOG_WARN("headers");
  }

void
Abstract_HTTP_Server_Decoder::
do_decode_http_entity()
  {
    POSEIDON_LOG_WARN("entity");
  }

void
Abstract_HTTP_Server_Decoder::
do_finish_http_message(HTTP_Decoder_State next)
  {
    POSEIDON_LOG_WARN("finish");
  }

void
Abstract_HTTP_Server_Decoder::
do_decode_websocket_frame()
  {
    POSEIDON_LOG_WARN("websocket");
  }

/*
bool
Abstract_HTTP_Server_Decoder::
do_finish_http_message(HTTP_Decoder_State next)
  {
    bool sent = this->do_http_server_send("", 0);

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
      sent = sent && this->do_http_server_send("0\r\n\r\n", 5);

    // If this is the final message, shut the connection down.
    this->m_state = next;
    if(this->m_final) {
      this->m_state = http_encoder_state_closed;
      sent = sent && this->do_http_server_close();
    }
    return sent;
  }


void
Abstract_HTTP_Server_Decoder::
do_decode_http_message_once()
  {
    // Try consuming some bytes.
    const char* bptr = this->m_queue.begin();
    const char* eptr = this->m_queue.end();
    if(bptr == eptr)
      return;



      size_t length;

      if(this->m_state == http_decoder_state_headers) {
        // Discard empty lines.
        // This makes sense if a client send an extra empty line after a POST body.
        length = 0;
        if(bptr[0] == '\n')
          length = 1;  // `\n`
        else if((eptr - bptr >= 2) && (bptr[0] == '\r') && (bptr[1] == '\n'))
          length = 2;  // `\r\n`

        this->m_queue.discard(length);
        if(length != 0)
          continue;

        // Search for two consecutive line breaks.
        while(++bptr != eptr) {
          if(bptr[0] != '\n')
            continue;

          // Here the string cannot start with `\n` or `\r\n`, so it is safe to read
          // backwards when a `\n` is encountered, without undeflowing the buffer.
          if(bptr[-1] == '\n')
            break;  // `\n\n`
          else if((bptr[-1] == '\r') && (bptr[-2] == '\n'))
            break;  // `\n\r\n`
        }

        // `bptr` shall point to the terminating line feed, or shall equal `eptr` if
        // the header is incomplete. In either case, data between `m_queue.begin()`
        // and `bptr` will be part of the headers, so check it.
        length = static_cast<size_t>(bptr - this->m_queue.begin()) + 1;
        if(length > this->m_max_header_length)
          POSEIDON_HTTP_THROW(
              http_status_headers_too_large,
              "HTTP headers too large (`$1` > `$2`)",
              length, this->m_max_header_length);

        // If the header is incomplete, stop.
        if(bptr == eptr)
          break;

        this->m_final = false;
        this->m_upgrading = false;
        this->m_chunked = false;
        this->m_gzip = false;

        this->m_content_offset = 0;
        this->m_content_length = 0;

        auto infl = unerase_pointer_cast<zlib_Inflator>(this->m_inflator);
        if(infl)
          infl->reset();

        // Parse headers as lines.
        HTTP_Method meth = http_method_null;
        cow_string target;
        HTTP_Version ver;
        Option_Map headers;

        cow_string line;
        
        bptr = this->m_queue.begin();

        for(;;) {
          // Get a UTF-8 code point.
          char32_t cp;
          if(!::asteria::utf8_decode(cp, bptr, static_cast<size_t>(eptr - bptr)))
            POSEIDON_THROW("HTTP headers not in UTF-8");

          // Ignore a `\r` that precedes a `\n`.
          // Valid data should be terminated by either `\n\n` or `\n\r\n`, so there
          // is no need to check for buffer overruns.
          if((cp == '\r') && (bptr[0] == '\n'))
            continue;

          if(cp != '\n') {
            // Check for control characters.
            if((cp == 0x7F) || ((cp != '\t') && (cp <= 0x1F)))
              POSEIDON_THROW("Control characters not allowed in HTTP headers");

            // Append it to the current line.
            line.push_back(static_cast<char>(cp));
            continue;
          }

          // Process a line. If the line is empty, finish.
          if(line.empty())
            break;

POSEIDON_LOG_FATAL("line: $1", line);

          if(meth == http_method_null) {
            // Parse the request line.
            //   request-line   = method SP request-target SP HTTP-version CRLF
            size_t ep_meth = line.find(' ');
            if(ep_meth == line.npos)
              POSEIDON_THROW("Invalid HTTP request line (missing target)");

            size_t ep_target = line.find(ep_meth + 1, ' ');
            if(ep_target == line.npos)
              POSEIDON_THROW("Invalid HTTP request line (missing HTTP version)");

            // Replace spaces with null characters.
            char* mptr = line.mut_data();
            mptr[ep_meth] = 0;
            mptr[ep_target] = 0;

            // Parse the HTTP version first so we don't mistake something not HTTP.
            ver = parse_http_version(mptr + ep_target + 1, mptr + line.size());
            if(ver == http_version_0_0)
              POSEIDON_THROW("Invalid HTTP request line (HTTP version not supported)");

            // Accept the method.
            meth = parse_http_method(mptr, mptr + ep_meth);
            if(meth == http_method_null)
              POSEIDON_HTTP_THROW(
                  http_status_not_implemented,
                  "HTTP method not recognized: $1",
                  mptr);

            // Accept the request target.
            target.assign(mptr + ep_meth + 1, mptr + ep_target);
          }
          else {
            // Parse a plain header.
            //   header-field   = field-name ":" OWS field-value OWS
this->m_queue.begin()            size_t ep_name = line.find(':');
            if(ep_name == line.npos)
              POSEIDON_HTTP_THROW(
                  http_status_bad_request,
                  "Malformed HTTP header: $1",
                  line);

            // Append this header.
            cow_string value(line, ep_name + 1);
            line.erase(ep_name);
            headers.append(line, ascii_trim(::std::move(value)));
          }
          line.clear();
        }

        ROCKET_ASSERT(bptr == eptr);
        this->m_queue.discard(header_length);

        HTTP_Connection conn = http_connection_keep_alive;
        bool proxy = target[0] != '/';
        bool no_content = false;
        Option_Map opts;

        auto connection_header_name = sref("Connection");
        if(proxy)
          connection_header_name = sref("Proxy-Connection");

        // Note if `Content-Length` is not specified, it is considered to have
        // a value of zero, which means there is no content.
        auto qstr = headers.find_opt(sref("Content-Length"));
        if(qstr) {
          // Parse the value as a signed 64-bit integer.
          ::rocket::ascii_numget numg;
          const char* sp = qstr->data();
          if(!numg.parse_U(sp, sp + qstr->size(), 10))
            POSEIDON_HTTP_THROW(
                http_status_bad_request,
                "Invalid `Content-Length` value: $1",
                *qstr);

          if(!numg.cast_U(this->m_content_length, 0, INT64_MAX))
            POSEIDON_HTTP_THROW(
                http_status_payload_too_large,
                "`Content-Length` value out of range: $1",
                *qstr);
        }

        // Parse `Transfer-Encoding`.
        cow_vstrings segs;
        qstr = headers.find_opt(sref("Transfer-Encoding"));
        if(qstr) {
          // HTTP/1.0 does not support it, but don't ignore it silently.
          if(ver < http_version_1_1)
            POSEIDON_HTTP_THROW(
                http_status_not_implemented,
                "`Transfer-Encoding` cannot be specified in HTTP/1.0");

          // Transfer encodings are delimited by commas.
          explode(segs, *qstr, ',');
        }

        // RFC 7230 doesn't specify that `identity` is a valid transfer encoding,
        // but Mozilla Developer Network has it as an example. For the sake of
        // compatibility, it is accepted but ignored, as if it wasn't specified.
        

        ??
        if((segs.size() == 1) && ascii_ci_equal(segs[0], sref("identity")))
          segs.clear();

        if(segs.empty()) {
          // This means `Content-Length` takes effect.
          no_content = this->m_content_length == 0;
        }
        else if((segs.size() == 1) && ascii_ci_equal(segs[0], sref("chunked"))) {
          // `chunked` means the entity is chunked.
          this->m_chunked = true;
        }
        else if((segs.size() == 2) && ascii_ci_equal(segs[0], sref("gzip")) &&
                                      ascii_ci_equal(segs[1], sref("chunked"))) {
          // `gzip,chunked` means the entity is chunked.
          this->m_chunked = true;
          this->m_gzip = true;
        }
        else
          POSEIDON_HTTP_THROW(
              http_status_not_implemented,
              "Unacceptable `Transfer-Encoding` value: $1",
              *qstr);

        // Get the request length and persistent connection state.
        if(ver < http_version_1_1)
          conn = http_connection_close;

        if(conn == http_connection_keep_alive)
          headers.for_each(connection_header_name,
              [&](const cow_string& str) {
                // XXX: Options may overwrite each other.
                if(ascii_ci_has_token(str, sref("keep-alive")))
                  conn = http_connection_keep_alive;

                if(ascii_ci_has_token(str, sref("close")))
                  conn = http_connection_close;
              });

        // Check for upgradable connections.
        if(conn == http_connection_upgrade) {
          this->m_final = true;
          this->m_upgrading = true;
        }

        // If the connection should be closed, say so.
        if(conn == http_connection_close)
          this->m_final = true;

        // Accept the headers.
        this->do_http_server_on_headers(meth, ::std::move(target), ver, ::std::move(headers));
        if(no_content) {
          this->do_finish_http_message(http_decoder_state_headers);
          continue;
        }
        
        






    if(no_content)
      return this->do_encode_http_headers(meth, target, ver, headers, conn) &&
             this->do_finish_http_message(http_encoder_state_headers);

    // Reset the deflator used by the previous message, if any.
    auto defl = unerase_pointer_cast<zlib_Deflator>(this->m_deflator);
    if(defl)
      defl->reset();

    // Expect the entity.
    bool sent = this->do_encode_http_headers(meth, target, ver, headers, conn);
    this->m_state = http_encoder_state_entity;
    return sent;
  }

        ? ::std::move(headers));

        // 


//??        this->m_state = http_decoder_state_headers;?
//        if(this->m_chunked || this->m_content_length)
//          this->m_state = http_decoder_state_entity;
        break;

        }
      }

      // If the connection has been closed, discard all buffered data.
      if(this->m_state == http_decoder_state_closed) {
        linear_buffer empty;
        this->m_queue.swap(empty);
        break;
      }
    }

*/





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

    // As it is possible for multiple messages to arrive simultaneously, we have
    // to try parsing repeatedly.
    this->m_queue.putn(data, size);

    for(;;) {
      const size_t old_size = this->m_queue.size();

      // Forward tunnel data verbatim.
      if(this->m_state == http_decoder_state_tunnel) {
        this->do_http_server_on_tunnel_data(this->m_queue.mut_data(), old_size);
        this->m_queue.clear();
        break;
      }

      // Each iteration shall consume some data; only when no progess can be made
      // shall the loop terminate.
      if(this->m_state == http_decoder_state_headers)
        this->do_decode_http_headers();

      if(this->m_state == http_decoder_state_entity)
        this->do_decode_http_entity();

      if(this->m_state == http_decoder_state_websocket)
        this->do_decode_websocket_frame();

      // When no progress can be made, finish.
      if(this->m_queue.size() == old_size)
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

    // Notify closure of the connection.
    if(this->m_state == http_decoder_state_tunnel)
      this->do_http_server_on_tunnel_closure();

    if(this->m_state == http_decoder_state_websocket)
      this->do_http_server_on_websocket_closure(
          websocket_status_no_status, reinterpret_cast<char*>(-1), 0);

    // Discard pending data and shut the connection down.
    this->m_state = http_decoder_state_closed;
    this->do_http_server_close();
    return true;
  }

}  // namespace poseidon
