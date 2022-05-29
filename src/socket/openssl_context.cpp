// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "openssl_context.hpp"
#include "../static/main_config.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"

namespace poseidon {

const OpenSSL_Context&
OpenSSL_Context::
static_verify_peer()
  {
    static uptr<OpenSSL_Context> s_ctx;
    static once_flag s_once;

    s_once.call(
      [] {
        s_ctx = ::rocket::make_unique<OpenSSL_Context>();

        // Enable certificate verification.
        ::SSL_CTX_set_verify(s_ctx->m_ctx, SSL_VERIFY_PEER, nullptr);
      });
    return *s_ctx;
  }

const OpenSSL_Context&
OpenSSL_Context::
static_verify_none()
  {
    static uptr<OpenSSL_Context> s_ctx;
    static once_flag s_once;

    s_once.call(
      [] {
        s_ctx = ::rocket::make_unique<OpenSSL_Context>();

        // Disable certificate verification.
        ::SSL_CTX_set_verify(s_ctx->m_ctx, SSL_VERIFY_NONE, nullptr);
      });
    return *s_ctx;
  }

OpenSSL_Context::
OpenSSL_Context()
  {
    this->m_ctx.reset(::SSL_CTX_new(::TLS_method()));
    if(!this->m_ctx)
      POSEIDON_SSL_THROW(
          "could not create SSL context\n"
          "[`SSL_CTX_new()` failed]");

    // Set CA path.
    // This should denote a directory containing trusted CA certificates.
    const auto file = Main_Config::copy();

    auto qstr = file.get_string_opt({"network","tls","trusted_ca_path"});
    if(qstr) {
      POSEIDON_LOG_INFO("Setting SSL CA certificate path to '$1'...", *qstr);
      if(::SSL_CTX_load_verify_locations(this->m_ctx, nullptr, qstr->safe_c_str()) != 1)
        POSEIDON_SSL_THROW(
            "could not set CA certificate path to '$1'\n"
            "[`SSL_CTX_set_default_verify_paths()` failed]",
            *qstr);
    }
    else
      POSEIDON_LOG_WARN(
          "CA certificate validation has been disabled.\n"
          "This configuration is not suitable for production use.\n"
          "Set `network.tls.trusted_ca_path` to enable.");

    // Set the session ID.
    // This is carried over processes, so we use a hard-coded string in
    // the executable.
    static constexpr uint32_t s_sid_len = SSL_MAX_SSL_SESSION_ID_LENGTH;
    static constexpr uint8_t s_sid_ctx[s_sid_len] = { PACKAGE_NAME __DATE__ __TIME__ };

    if(::SSL_CTX_set_session_id_context(this->m_ctx, s_sid_ctx, s_sid_len) != 1)
      POSEIDON_SSL_THROW(
          "could not set SSL session id context\n"
          "[`SSL_CTX_set_session_id_context()` failed]");
  }

OpenSSL_Context::
~OpenSSL_Context()
  {
  }

}  // namespace poseidon
