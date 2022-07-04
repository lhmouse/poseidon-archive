// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "network_driver.hpp"
#include "async_logger.hpp"
#include "../socket/abstract_socket.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"
#include <sys/epoll.h>
#include <openssl/ssl.h>
#include <openssl/err.h>

namespace poseidon {
namespace {

void
do_epoll_ctl(int epoll_fd, int op, const shared_ptr<Abstract_Socket>& socket, uint32_t events)
  {
    struct ::epoll_event event;
    event.events = events;
    event.data.ptr = socket.get();
    if(::epoll_ctl(epoll_fd, op, socket->fd(), &event) != 0)
      POSEIDON_LOG_ERROR((
          "Could not modify socket `$2` (class `$3`)",
          "[`epoll_ctl()` failed: $1]"),
          format_errno(), socket, typeid(*socket));

    if((op == EPOLL_CTL_ADD) || (op == EPOLL_CTL_MOD))
      POSEIDON_LOG_TRACE((
          "Updated epoll flags for socket `$1` (class `$2`): ET = $3, IN = $4, OUT = $5"),
          socket, typeid(*socket), (event.events / EPOLLET) & 1U, (event.events / EPOLLIN) & 1U,
          (event.events / EPOLLOUT) & 1U);
  }

}  // namespace

Network_Driver::
Network_Driver()
  {
    // Allocate epoll objects.
    this->m_epoll_lt.reset(::epoll_create1(0));
    if(!this->m_epoll_lt)
      POSEIDON_THROW((
          "Failed to allocate epoll descriptor (LT)",
          "[`epoll_create1()` failed: $1]"),
          format_errno());

    this->m_epoll_et.reset(::epoll_create1(0));
    if(!this->m_epoll_et)
      POSEIDON_THROW((
          "Failed to allocate epoll descriptor (ET)",
          "[`epoll_create1()` failed: $1]"),
          format_errno());
  }

Network_Driver::
~Network_Driver()
  {
  }

SSL_CTX_ptr
Network_Driver::
default_server_ssl_ctx() const
  {
    plain_mutex::unique_lock lock(this->m_conf_mutex);

    if(!this->m_server_ssl_ctx)
      POSEIDON_LOG_WARN((
          "Server SSL context unavailable",
          "[certificate not configured in 'main.conf']"));

    return this->m_server_ssl_ctx;
  }

SSL_CTX_ptr
Network_Driver::
default_client_ssl_ctx() const
  {
    plain_mutex::unique_lock lock(this->m_conf_mutex);

    if(!this->m_client_ssl_ctx)
      POSEIDON_LOG_WARN((
          "Client SSL context unavailable",
          "[no configuration loaded]"));

    return this->m_client_ssl_ctx;
  }

void
Network_Driver::
reload(const Config_File& file)
  {
    // Parse new configuration. Default ones are defined here.
    int64_t event_buffer_size = 1024;
    int64_t throttle_size = 1048576;
    cow_string default_certificate;
    cow_string default_private_key;
    cow_string trusted_ca_path;

    SSL_CTX_ptr server_ssl_ctx;
    SSL_CTX_ptr client_ssl_ctx;

    // Read the event buffer size from configuration.
    auto value = file.query("network", "poll", "event_buffer_size");
    if(value.is_integer())
      event_buffer_size = value.as_integer();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `network.poll.event_buffer_size`: expecting an `integer`, got `$1`",
          "[in configuration file '$2']"),
          value, file.path());

    if((event_buffer_size < 0x10) || (event_buffer_size > 0x7FFFF0))
      POSEIDON_THROW((
          "`network.poll.event_buffer_size` value `$1` out of range",
          "[in configuration file '$2']"),
          event_buffer_size, file.path());

    // Read the throttle size from configuration.
    value = file.query("network", "poll", "throttle_size");
    if(value.is_integer())
      throttle_size = value.as_integer();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `network.poll.throttle_size`: expecting an `integer`, got `$1`",
          "[in configuration file '$2']"),
          value, file.path());

    if((throttle_size < 0x100) || (throttle_size > 0x7FFFFFF0))
      POSEIDON_THROW((
          "`network.poll.throttle_size` value `$1` out of range",
          "[in configuration file '$2']"),
          throttle_size, file.path());

    // Get the path to the default server certificate and private key.
    value = file.query("network", "ssl", "default_certificate");
    if(value.is_string())
      default_certificate = value.as_string();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `network.ssl.default_certificate`: expecting a `string`, got `$1`",
          "[in configuration file '$2']"),
          value, file.path());

    value = file.query("network", "ssl", "default_private_key");
    if(value.is_string())
      default_private_key = value.as_string();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `network.ssl.default_private_key`: expecting a `string`, got `$1`",
          "[in configuration file '$2']"),
          value, file.path());

    // Check for configuration errors.
    if(!default_certificate.empty() && default_private_key.empty())
      POSEIDON_THROW((
          "`network.ssl.default_private_key` missing",
          "[in configuration file '$1']"),
          file.path());

    if(default_certificate.empty() && !default_private_key.empty())
      POSEIDON_THROW((
          "`network.ssl.default_private_key` missing",
          "[in configuration file '$1']"),
          file.path());

    if(!default_certificate.empty() && !default_private_key.empty()) {
      // Create the server context.
      server_ssl_ctx.reset(::SSL_CTX_new(::TLS_server_method()));
      if(!server_ssl_ctx)
        POSEIDON_THROW((
            "Could not allocate server SSL context",
            "[`SSL_CTX_new()` failed]: $1"),
            ::ERR_reason_error_string(::ERR_peek_error()));

      // Load the certificate and private key.
      if(!::SSL_CTX_use_certificate_chain_file(server_ssl_ctx, default_certificate.safe_c_str()))
        POSEIDON_THROW((
            "Could not load default server SSL certificate file '$3'",
            "[`SSL_CTX_use_certificate_chain_file()` failed: $1]",
            "[in configuration file '$2']"),
            ::ERR_reason_error_string(::ERR_peek_error()), file.path(), default_certificate);

      if(!::SSL_CTX_use_PrivateKey_file(server_ssl_ctx, default_private_key.safe_c_str(), SSL_FILETYPE_PEM))
        POSEIDON_THROW((
            "Could not load default server SSL private key file '$3'",
            "[`SSL_CTX_use_PrivateKey_file()` failed: $1]",
            "[in configuration file '$2']"),
            ::ERR_reason_error_string(::ERR_peek_error()), file.path(), default_private_key);

      if(!::SSL_CTX_check_private_key(server_ssl_ctx))
        POSEIDON_THROW((
            "Error validating default server SSL certificate '$3' and SSL private key '$4'",
            "[`SSL_CTX_check_private_key()` failed: $1]",
            "[in configuration file '$2']"),
            ::ERR_reason_error_string(::ERR_peek_error()), file.path(), default_certificate, default_private_key);

      ::SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER | SSL_VERIFY_CLIENT_ONCE, nullptr);
    }

    // Get the path to trusted CA certificates.
    value = file.query("network", "ssl", "trusted_ca_path");
    if(value.is_string())
      trusted_ca_path = value.as_string();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `network.ssl.trusted_ca_path`: expecting a `string`, got `$1`",
          "[in configuration file '$2']"),
          value, file.path());

    // Create the client context, always.
    client_ssl_ctx.reset(::SSL_CTX_new(::TLS_client_method()));
    if(!client_ssl_ctx)
      POSEIDON_THROW((
          "Could not allocate client SSL context: $1",
          "[`SSL_CTX_new()` failed]"),
          ::ERR_reason_error_string(::ERR_peek_error()));

    if(!trusted_ca_path.empty()) {
      // Load trusted CA certificates from the given directory.
      if(!::SSL_CTX_load_verify_locations(client_ssl_ctx, nullptr, trusted_ca_path.safe_c_str()))
        POSEIDON_THROW((
            "Could not set path to trusted CA certificates",
            "[`SSL_CTX_load_verify_locations()` failed: $1]",
            "[in configuration file '$2']"),
            ::ERR_reason_error_string(::ERR_peek_error()), file.path());

      // Use the hostname as the context.
      char hostname[SSL_MAX_SID_CTX_LENGTH] = { };
      ::gethostname(hostname, sizeof(hostname));

      if(!::SSL_CTX_set_session_id_context(client_ssl_ctx, (unsigned char*) hostname, sizeof(hostname)))
        POSEIDON_THROW((
            "Could not set SSL session ID context",
            "[`SSL_set_session_id_context()` failed: $1]",
            "[in configuration file '$2']"),
            ::ERR_reason_error_string(::ERR_peek_error()), file.path());

      ::SSL_CTX_set_verify(client_ssl_ctx, SSL_VERIFY_PEER, nullptr);
    }
    else {
      POSEIDON_LOG_WARN((
          "CA certificate validation has been disabled. This configuration is not "
          "recommended for production use.",
          "Set `network.ssl.trusted_ca_path` in '$1' to enable it."),
          file.path());

      ::SSL_CTX_set_verify(client_ssl_ctx, SSL_VERIFY_NONE, nullptr);
    }

    // Set up new data.
    plain_mutex::unique_lock lock(this->m_conf_mutex);
    this->m_event_buffer_size = (uint32_t) event_buffer_size;
    this->m_throttle_size = (uint32_t) throttle_size;
    this->m_server_ssl_ctx.swap(server_ssl_ctx);
    this->m_client_ssl_ctx.swap(client_ssl_ctx);
  }

void
Network_Driver::
thread_loop()
  {
    // Await events.
    shared_ptr<Abstract_Socket> socket;
    struct ::epoll_event event;

    plain_mutex::unique_lock lock(this->m_conf_mutex);
    const size_t event_buffer_size = this->m_event_buffer_size;
    const size_t throttle_size = this->m_throttle_size;
    lock.unlock();

    // Get an event from the event queue. If the queue has been exhausted, reload
    // some from epolls. The level-triggered epoll indicates purely incoming data,
    // while the edge-triggered epoll indicates both incoming and outgoing data.
    // When there is no event pending, the network thread should only wait for the
    // level-triggered epoll.
    lock.lock(this->m_event_mutex);
    if(this->m_events.getn((char*) &event, sizeof(event)) == 0) {
      this->m_events.reserve(sizeof(::epoll_event) * event_buffer_size);
      ::epoll_event* const event_buffer = (::epoll_event*) this->m_events.end();

      int ngot = ::epoll_wait(this->m_epoll_lt, event_buffer, (int) event_buffer_size, 0);
      if(ngot <= 0) {
        // Collect events from the edge-triggered epoll.
        ngot = ::epoll_wait(this->m_epoll_et, event_buffer, (int) event_buffer_size, 5000);
        if(ngot <= 0)
          return;

        // Because `::epoll_wait()` does not report `EPOLLET`, it has to be added
        // by hand. This is necessary for telling where the events come from.
        for(uint32_t k = 0;  k < (uint32_t) ngot;  ++k)
          event_buffer[k].events |= EPOLLET;
      }
      this->m_events.accept(sizeof(::epoll_event) * (uint32_t) ngot);
      POSEIDON_LOG_TRACE(("Collected `$1` socket event(s) from epoll"), (uint32_t) ngot);

      this->m_events.getn((char*) &event, sizeof(event));
    }
    lock.unlock();

    // Get the socket.
    lock.lock(this->m_epoll_mutex);
    auto socket_it = this->m_epoll_sockets.find(event.data.ptr);
    if(socket_it == this->m_epoll_sockets.end())
      return;

    socket = socket_it->second.lock();
    if(!socket) {
      // Remove expired sockete. It will be deleted automatically after being closed.
      this->m_epoll_sockets.erase(socket_it);
      POSEIDON_LOG_TRACE(("Socket expired: $1"), event.data.ptr);
      return;
    }
    else if(event.events & (EPOLLHUP | EPOLLERR)) {
      // Remove the socket due to an error, or an end-of-file condition.
      POSEIDON_LOG_TRACE(("Removing closed socket `$1` (class `$2`)"), socket, typeid(*socket));
      this->m_epoll_sockets.erase(socket_it);
      do_epoll_ctl(this->m_epoll_lt, EPOLL_CTL_DEL, socket, 0);
      do_epoll_ctl(this->m_epoll_et, EPOLL_CTL_DEL, socket, 0);
    }
    recursive_mutex::unique_lock io_lock(socket->m_io_mutex);
    socket->m_io_driver = this;
    lock.unlock();

    // Process events on this socket.
    POSEIDON_LOG_TRACE((
        "Processing socket `$1` (class `$2`): ET = $3, HUP = $4, ERR = $5, IN = $6, OUT = $7"),
        socket, typeid(*socket), (event.events / EPOLLET) & 1U, (event.events / EPOLLHUP) & 1U,
        (event.events / EPOLLERR) & 1U, (event.events / EPOLLIN) & 1U, (event.events / EPOLLOUT) & 1U);

    if(event.events & (EPOLLHUP | EPOLLERR)) {
      // Get its error code, if an error has been reported.
      int err = 0;
      if(event.events & EPOLLERR) {
        ::socklen_t optlen = sizeof(err);
        if(::getsockopt(socket->fd(), SOL_SOCKET, SO_ERROR, &err, &optlen) != 0)
          err = errno;
      }

      // Deliver the closure notification.
      POSEIDON_LOG_DEBUG(("Socket `$1` (class `$2`) closed: $3"), socket, typeid(*socket), format_errno(err));
      socket->m_state.store(socket_state_closed);

      try {
        socket->do_abstract_socket_on_closed(err);
      }
      catch(exception& stdex) {
        POSEIDON_LOG_ERROR((
            "Unhandled exception thrown from socket closure callback: $1",
            "[socket class `$2`]"),
            stdex, typeid(*socket));
      }

      // Exit now. This socket shall have been removed from this epoll.
      socket->m_state.store(socket_state_closed);
      return;
    }

    if(socket->m_state.load() == socket_state_closed) {
      // Force closure.
      ::shutdown(socket->fd(), SHUT_RDWR);
      POSEIDON_LOG_TRACE(("Socket `$1` (class `$2`) shutdown pending"), socket, typeid(*socket));
      return;
    }

    if(event.events & EPOLLIN) {
      // Deliver the readability notification.
      POSEIDON_LOG_TRACE(("Socket `$1` (class `$2`) readable"), socket, typeid(*socket));
      socket->m_state.store(socket_state_established);

      try {
        socket->do_abstract_socket_on_readable();
      }
      catch(exception& stdex) {
        POSEIDON_LOG_ERROR((
            "Unhandled exception thrown from socket read callback: $1",
            "[socket class `$2`]"),
            stdex, typeid(*socket));

        socket->do_abstract_socket_on_exception(stdex);
      }

      // If there are too many bytes pending, disable `EPOLLIN` notification.
      if(socket->m_io_write_queue.size() >= throttle_size)
        do_epoll_ctl(this->m_epoll_lt, EPOLL_CTL_MOD, socket, EPOLLOUT);
    }

    if(event.events & EPOLLOUT) {
      // Deliver the writability notification.
      POSEIDON_LOG_TRACE(("Socket `$1` (class `$2`) writable"), socket, typeid(*socket));
      socket->m_state.store(socket_state_established);

      try {
        socket->do_abstract_socket_on_writable();
      }
      catch(exception& stdex) {
        POSEIDON_LOG_ERROR((
            "Unhandled exception thrown from socket write callback: $1",
            "[socket class `$2`]"),
            stdex, typeid(*socket));

        socket->do_abstract_socket_on_exception(stdex);
      }

      // If this event came from the level-triggered epoll, and there are fewer
      // bytes pending, re-enable `EPOLLIN` notification.
      if(!(event.events & EPOLLET) && (socket->m_io_write_queue.size() < throttle_size))
        do_epoll_ctl(this->m_epoll_lt, EPOLL_CTL_MOD, socket, EPOLLIN);
    }

    POSEIDON_LOG_TRACE(("Socket `$1` (class `$2`) I/O complete"), socket, typeid(*socket));
    socket->m_io_driver = (Network_Driver*) 123456789;
  }

void
Network_Driver::
insert(const shared_ptr<Abstract_Socket>& socket)
  {
    // Validate arguments.
    if(!socket)
      POSEIDON_THROW(("Null socket pointer not valid"));

    // Register the socket. Note exception safety.
    // The socket will be deleted from an epoll automatically when it's closed,
    // so there is no need to remove it in case of an exception.
    plain_mutex::unique_lock lock(this->m_epoll_mutex);
    do_epoll_ctl(this->m_epoll_lt, EPOLL_CTL_ADD, socket, EPOLLIN);
    do_epoll_ctl(this->m_epoll_et, EPOLL_CTL_ADD, socket, EPOLLIN | EPOLLOUT | EPOLLET);
    this->m_epoll_sockets[socket.get()] = socket;
  }

}  // namespace poseidon
