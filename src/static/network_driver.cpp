// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "network_driver.hpp"
#include "main_config.hpp"
#include "../core/config_file.hpp"
#include "../socket/abstract_socket.hpp"
#include "../utils.hpp"
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/eventfd.h>
#include <signal.h>

namespace poseidon {
namespace {

struct Config_Scalars
  {
    size_t event_buffer_size = 1024;
    size_t throttle_size = 1048576;
  };

enum : uint32_t
  {
    poll_index_max    = 0x00FFFFF0,  // 24 bits
    poll_index_event  = 0x00FFFFF1,  // index for the eventfd
    poll_index_end    = 0xFFFFFFFE,  // end of list
    poll_index_nil    = 0xFFFFFFFF,  // bad position
  };

struct Poll_List_mixin
  {
    uint32_t next = poll_index_nil;
    uint32_t prev = poll_index_nil;
  };

struct Poll_Socket
  {
    rcptr<Abstract_Socket> sock;
    Poll_List_mixin node_cl;  // closed
    Poll_List_mixin node_rd;  // readable
    Poll_List_mixin node_wr;  // writable
  };

template<Poll_List_mixin Poll_Socket::* mptrT>
struct Poll_List_root
  {
    uint32_t head = poll_index_end;
    uint32_t tail = poll_index_end;
  };

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Network_Driver)
  {
    // constant data
    once_flag m_init_once;
    ::pthread_t m_thread;
    int m_epoll_fd = -1;
    int m_event_fd = -1;

    // configuration
    mutable simple_mutex m_conf_mutex;
    Config_Scalars m_conf;

    // dynamic data
    ::std::vector<::epoll_event> m_event_buffer;
    ::std::vector<rcptr<Abstract_Socket>> m_ready_socks;

    mutable simple_mutex m_poll_mutex;
    ::std::vector<Poll_Socket> m_poll_elems;
    uint64_t m_poll_serial = 0;
    Poll_List_root<&Poll_Socket::node_cl> m_poll_root_cl;
    Poll_List_root<&Poll_Socket::node_rd> m_poll_root_rd;
    Poll_List_root<&Poll_Socket::node_wr> m_poll_root_wr;

    static
    void
    do_start()
      {
        self->m_init_once.call(
          [&] {
            // Create an epoll object.
            unique_FD epoll_fd(::epoll_create(100));
            if(!epoll_fd)
              POSEIDON_THROW(
                  "could not create epoll object\n"
                  "[`epoll_create()` failed: $1]",
                  format_errno());

            // Create the notification eventfd and add it into epoll.
            unique_FD event_fd(::eventfd(0, EFD_NONBLOCK));
            if(!event_fd)
              POSEIDON_THROW(
                  "could not create eventfd object\n"
                  "[`eventfd()` failed: $1]",
                  format_errno());

            ::epoll_event event;
            event.data.u64 = self->make_epoll_data(poll_index_event, 0);
            event.events = EPOLLIN | EPOLLET;
            if(::epoll_ctl(epoll_fd, EPOLL_CTL_ADD, event_fd, &event) != 0)
              POSEIDON_THROW(
                  "failed to add socket into epoll\n"
                  "[`epoll_ctl()` failed: $1]",
                  format_errno());

            // Create the thread. Note it is never joined or detached.
            simple_mutex::unique_lock lock(self->m_conf_mutex);
            int err = ::pthread_create(&(self->m_thread), nullptr, do_thread_procedure, nullptr);
            if(err != 0)
              POSEIDON_THROW("`pthread_create()` failed: $1", format_errno(err));

            self->m_epoll_fd = epoll_fd.release();
            self->m_event_fd = event_fd.release();
          });
      }

    [[noreturn]] static
    void*
    do_thread_procedure(void*)
      {
        // Set thread information. Errors are ignored.
        ::sigset_t sigset;
        ::sigemptyset(&sigset);
        ::sigaddset(&sigset, SIGINT);
        ::sigaddset(&sigset, SIGTERM);
        ::sigaddset(&sigset, SIGHUP);
        ::sigaddset(&sigset, SIGALRM);
        ::pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

        int oldst;
        ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &oldst);

        ::pthread_setname_np(::pthread_self(), "network");

        // Enter an infinite loop.
        for(;;)
          try {
            self->do_thread_loop();
          }
          catch(exception& stdex) {
            POSEIDON_LOG_FATAL(
                "Caught an exception from network thread loop: $1\n"
                "[exception class `$2`]\n",
                stdex.what(), typeid(stdex).name());
          }
      }

    // The index takes up 24 bits. That's 16M simultaneous connections.
    // The serial number takes up 40 bits. That's ~1.1T historical connections.
    static constexpr
    uint64_t
    make_epoll_data(uint64_t index, uint64_t serial) noexcept
      {
        return (index << 40) | ((serial << 24) >> 24);
      }

    static constexpr
    uint32_t
    index_from_epoll_data(uint64_t epoll_data) noexcept
      {
        return static_cast<uint32_t>(epoll_data >> 40);
      }

    // Note epoll events are bound to kernel files, not individual file descriptors.
    // If we were passing pointers in `event.data` and FDs got `dup()`'d, we could get
    // dangling pointers here, which is rather dangerous.
    static
    uint32_t
    find_poll_socket(uint64_t epoll_data) noexcept
      {
        uint32_t index = self->index_from_epoll_data(epoll_data);
        if(ROCKET_EXPECT(index < self->m_poll_elems.size())) {
          // Check whether the hint has been invalidated.
          const auto& elem = self->m_poll_elems[index];
          if(ROCKET_EXPECT(elem.sock->m_epoll_data == epoll_data))
            return index;
        }

        // Perform a brute-force search.
        for(index = 0;  index < self->m_poll_elems.size();  ++index) {
          const auto& elem = self->m_poll_elems[index];
          if(ROCKET_UNEXPECT(elem.sock->m_epoll_data == epoll_data)) {
            // Update the lookup hint. Errors are ignored.
            ::epoll_event event;
            event.data.u64 = self->make_epoll_data(index, epoll_data);
            event.events = EPOLLIN | EPOLLOUT | EPOLLET;
            if(::epoll_ctl(self->m_epoll_fd, EPOLL_CTL_MOD, elem.sock->get_fd(), &event) != 0)
              POSEIDON_LOG_FATAL(
                  "Failed to modify socket in epoll\n"
                  "[`epoll_ctl()` failed: $1]",
                  format_errno());

            elem.sock->m_epoll_data = event.data.u64;
            POSEIDON_LOG_DEBUG("Epoll lookup hint updated: value = $1", event.data.u64);

            // Return the new index.
            ROCKET_ASSERT(index != poll_index_nil);
            return index;
          }
        }

        // This should not happen at all.
        POSEIDON_LOG_FATAL("Socket not found: epoll_data = $1", epoll_data);
        return poll_index_nil;
      }

    ROCKET_PURE static
    bool
    poll_lists_empty() noexcept
      {
        return (self->m_poll_root_cl.head == poll_index_end) &&  // close list empty
               (self->m_poll_root_rd.head == poll_index_end) &&  // read list empty
               (self->m_poll_root_wr.head == poll_index_end);    // write list empty
      }

    template<Poll_List_mixin Poll_Socket::* mptrT>
    static
    size_t
    poll_list_collect(const Poll_List_root<mptrT>& root)
      {
        self->m_ready_socks.clear();

        // Iterate over all elements and push them all.
        uint32_t index = root.head;
        while(index != poll_index_end) {
          ROCKET_ASSERT(index != poll_index_nil);
          const auto& elem = self->m_poll_elems[index];
          index = (elem.*mptrT).next;
          self->m_ready_socks.emplace_back(elem.sock);
        }
        return self->m_ready_socks.size();
      }

    template<Poll_List_mixin Poll_Socket::* mptrT>
    static
    bool
    poll_list_attach(Poll_List_root<mptrT>& root, uint32_t index) noexcept
      {
        // Don't perform any operation if the element has already been attached.
        auto& elem = self->m_poll_elems[index];
        if((elem.*mptrT).next != poll_index_nil)
          return false;

        // Insert this node at the end of the doubly linked list.
        uint32_t prev = ::std::exchange(root.tail, index);
        ((prev != poll_index_end) ? (self->m_poll_elems[prev].*mptrT).next : root.head) = index;
        (elem.*mptrT).next = poll_index_end;
        (elem.*mptrT).prev = prev;
        return true;
      }

    template<Poll_List_mixin Poll_Socket::* mptrT>
    static
    bool
    poll_list_detach(Poll_List_root<mptrT>& root, uint32_t index) noexcept
      {
        // Don't perform any operation if the element has not been attached.
        auto& elem = self->m_poll_elems[index];
        if((elem.*mptrT).next == poll_index_nil)
          return false;

        // Remove this node from the doubly linked list.
        uint32_t next = ::std::exchange((elem.*mptrT).next, poll_index_nil);
        uint32_t prev = ::std::exchange((elem.*mptrT).prev, poll_index_nil);
        ((next != poll_index_end) ? (self->m_poll_elems[next].*mptrT).prev : root.tail) = prev;
        ((prev != poll_index_end) ? (self->m_poll_elems[prev].*mptrT).next : root.head) = next;
        return true;
      }

    static
    void
    do_thread_loop()
      {
        // Reload configuration.
        simple_mutex::unique_lock lock(self->m_conf_mutex);
        const auto conf = self->m_conf;
        lock.unlock();

        self->m_event_buffer.resize(conf.event_buffer_size);
        self->m_ready_socks.clear();

        // Try polling if there is nothing to do.
        lock.lock(self->m_poll_mutex);
        if(self->poll_lists_empty()) {
          for(const auto& elem : self->m_poll_elems) {
            if(elem.sock.unique() && !elem.sock->m_resident.load()) {
              // Delete sockets that have no other references to them.
              POSEIDON_LOG_DEBUG("Killed orphan socket: $1", elem.sock);
              elem.sock->kill();
              continue;
            }
            POSEIDON_LOG_TRACE("Active socket: $1", elem.sock);
          }
          lock.unlock();

          // Await I/O events.
          int navail = ::epoll_wait(self->m_epoll_fd, self->m_event_buffer.data(),
                                    static_cast<int>(self->m_event_buffer.size()),
                                    60'000);  // one minute
          if(navail < 0) {
            POSEIDON_LOG_TRACE("`epoll_wait()` failed: $1", format_errno());
            self->m_event_buffer.clear();
          }
          else
            self->m_event_buffer.resize(static_cast<uint32_t>(navail));

          // Process all events that have been received so far.
          // Note the loop below will not throw exceptions.
          lock.lock(self->m_poll_mutex);
          for(const auto& event : self->m_event_buffer) {
            // Check for special indexes.
            if(self->index_from_epoll_data(event.data.u64) == poll_index_event) {
              // There was an explicit wakeup so make the event non-signaled.
              ::eventfd_t ignored;
              while(::eventfd_read(self->m_event_fd, &ignored) != 0) {
                int err = errno;
                if(::rocket::is_none_of(err, { EINTR, EAGAIN, EWOULDBLOCK })) {
                  POSEIDON_LOG_FATAL("`eventfd_read()` failed: $1", format_errno(err));
                  break;
                }
                else if(err != EINTR)
                  break;
              }
              continue;
            }

            // Find the socket.
            uint32_t index = self->find_poll_socket(event.data.u64);
            if(index == poll_index_nil)
              continue;

            // Update socket event flags.
            const auto& elem = self->m_poll_elems[index];
            elem.sock->m_epoll_events |= event.events;

            // Update close/read/write lists.
            if(event.events & (EPOLLERR | EPOLLHUP))
              self->poll_list_attach(self->m_poll_root_cl, index);
            if(event.events & EPOLLIN)
              self->poll_list_attach(self->m_poll_root_rd, index);
            if(event.events & EPOLLOUT)
              self->poll_list_attach(self->m_poll_root_wr, index);
          }
        }

        // Process closed sockets.
        lock.lock(self->m_poll_mutex);
        self->poll_list_collect(self->m_poll_root_cl);
        for(const auto& sock : self->m_ready_socks) {
          lock.lock(self->m_poll_mutex);
          int err = sock->m_epoll_events & EPOLLERR;
          lock.unlock();

          // Get the error number if `EPOLLERR` has been turned on.
          if(err) {
            ::socklen_t optlen = sizeof(err);
            if(::getsockopt(sock->get_fd(), SOL_SOCKET, SO_ERROR, &err, &optlen) != 0)
              err = errno;
          }
          POSEIDON_LOG_TRACE("Socket closed: $1 ($2)", sock, format_errno(err));

          // Remove the socket from epoll. Errors are ignored.
          if(::epoll_ctl(self->m_epoll_fd, EPOLL_CTL_DEL, sock->get_fd(), nullptr) != 0)
            POSEIDON_LOG_FATAL(
                "failed to remove socket from epoll\n"
                "[`epoll_ctl()` failed: $1]",
                format_errno());

          try {
            sock->do_socket_on_poll_close(err);
          }
          catch(exception& stdex) {
            POSEIDON_LOG_WARN(
                "Socket close error: $1\n"
                "[socket class `$2`]",
                stdex.what(), typeid(*sock));
          }

          // Remove the socket, no matter whether an exception was thrown or not.
          lock.lock(self->m_poll_mutex);
          uint32_t index = self->find_poll_socket(sock->m_epoll_data);
          if(index == poll_index_nil)
            continue;

          self->poll_list_detach(self->m_poll_root_cl, index);
          self->poll_list_detach(self->m_poll_root_rd, index);
          self->poll_list_detach(self->m_poll_root_wr, index);

          // Swap the socket with the last element for removal.
          uint32_t ilast = static_cast<uint32_t>(self->m_poll_elems.size() - 1);
          if(index != ilast) {
            swap(self->m_poll_elems[ilast].sock,
              self->m_poll_elems[index].sock);

            if(self->poll_list_detach(self->m_poll_root_cl, ilast))
              self->poll_list_attach(self->m_poll_root_cl, index);
            if(self->poll_list_detach(self->m_poll_root_rd, ilast))
              self->poll_list_attach(self->m_poll_root_rd, index);
            if(self->poll_list_detach(self->m_poll_root_wr, ilast))
              self->poll_list_attach(self->m_poll_root_wr, index);
          }
          self->m_poll_elems.pop_back();
          POSEIDON_LOG_TRACE("Removed closed socket: $1", sock);
        }

        // Process readable sockets.
        lock.lock(self->m_poll_mutex);
        self->poll_list_collect(self->m_poll_root_rd);
        for(const auto& sock : self->m_ready_socks) {
          lock.unlock();

          bool detach;
          bool clear_status;

          try {
            if(sock->do_write_queue_size(lock) > conf.throttle_size) {
              // If the socket is throttled, remove it from read queue.
              detach = true;
              clear_status = false;
            }
            else {
              // Perform a single read operation (no retry upon EINTR).
              auto io_res = sock->do_socket_on_poll_read(lock);

              // If the read operation didn't proceed, the socket shall be removed from
              // read queue and the `EPOLLIN` status shall be cleared.
              detach = io_res != io_result_partial_work;
              clear_status = io_res != io_result_partial_work;
            }
          }
          catch(exception& stdex) {
            POSEIDON_LOG_WARN(
                "Socket read error: $1\n"
                "[socket class `$2`]", stdex.what(), typeid(*sock));

            // Force closure of the connection.
            sock->kill();

            // If a read error occurs, the socket shall be removed from read queue and
            // the `EPOLLIN` status shall be cleared.
            detach = true;
            clear_status = true;
          }

          // Update the socket.
          lock.lock(self->m_poll_mutex);
          uint32_t index = self->find_poll_socket(sock->m_epoll_data);
          if(index == poll_index_nil)
            continue;

          if(detach)
            self->poll_list_detach(self->m_poll_root_rd, index);
          if(clear_status)
            sock->m_epoll_events &= ~EPOLLIN;
        }

        // Process writable sockets.
        lock.lock(self->m_poll_mutex);
        self->poll_list_collect(self->m_poll_root_wr);
        for(const auto& sock : self->m_ready_socks) {
          lock.unlock();

          bool unthrottle;
          bool detach;
          bool clear_status;

          try {
            // Perform a single write operation (no retry upon EINTR).
            auto io_res = sock->do_socket_on_poll_write(lock);

            // Check whether the socket should be unthrottled.
            unthrottle = (sock->m_epoll_events & EPOLLIN) &&
                         (sock->do_write_queue_size(lock) <= conf.throttle_size);

            // If the write operation didn't proceed, the socket shall be removed from
            // write queue. If the write operation reported `io_result_would_block`, in
            // addition to the removal, the `EPOLLOUT` status shall be cleared.
            detach = io_res != io_result_partial_work;
            clear_status = io_res == io_result_would_block;
          }
          catch(exception& stdex) {
            POSEIDON_LOG_WARN(
                "Socket write error: $1\n"
                "[socket class `$2`]", stdex.what(), typeid(*sock));

            // Force closure of the connection.
            sock->kill();
            unthrottle = false;

            // If a write error occurs, the socket shall be removed from write queue and
            // the `EPOLLOUT` status shall be cleared.
            detach = true;
            clear_status = true;
          }

          // Update the socket.
          lock.lock(self->m_poll_mutex);
          uint32_t index = self->find_poll_socket(sock->m_epoll_data);
          if(index == poll_index_nil)
            continue;

          if(unthrottle)
            self->poll_list_attach(self->m_poll_root_rd, index);
          if(detach)
            self->poll_list_detach(self->m_poll_root_wr, index);
          if(clear_status)
            sock->m_epoll_events &= ~EPOLLOUT;
        }
      }

    static
    bool
    do_signal_if_poll_lists_empty() noexcept
      {
        if(ROCKET_EXPECT(!self->poll_lists_empty()))
          return false;

        // Make the event signaled.
        while(::eventfd_write(self->m_event_fd, 1) != 0) {
          int err = errno;
          if(::rocket::is_none_of(err, { EINTR, EAGAIN, EWOULDBLOCK })) {
            POSEIDON_LOG_FATAL("`eventfd_write()` failed: $1", format_errno(err));
            break;
          }
          else if(err != EINTR)
            break;
        }
        return true;
      }
  };

void
Network_Driver::
start()
  {
    self->do_start();
  }

void
Network_Driver::
reload()
  {
    // Load network settings into temporary objects.
    const auto file = Main_Config::copy();
    Config_Scalars conf;

    auto qint = file.get_int64_opt({"network","poll","event_buffer_size"});
    if(qint)
      conf.event_buffer_size = clamp_cast<size_t>(*qint, 1, 4096);

    qint = file.get_int64_opt({"network","poll","throttle_size"});
    if(qint)
      conf.throttle_size = clamp_cast<size_t>(*qint, 1, 1048576);

    // During destruction of temporary objects the mutex should have been unlocked.
    // The swap operation is presumed to be fast, so we don't hold the mutex
    // for too long.
    simple_mutex::unique_lock lock(self->m_conf_mutex);
    self->m_conf = ::std::move(conf);
  }

rcptr<Abstract_Socket>
Network_Driver::
insert(uptr<Abstract_Socket>&& usock)
  {
    // Take ownership of `usock`.
    rcptr<Abstract_Socket> sock(usock.release());
    if(!sock)
      POSEIDON_THROW("null socket pointer not valid");

    if(!sock.unique())
      POSEIDON_THROW("socket pointer must be unique");

    // Lock epoll for modification.
    simple_mutex::unique_lock lock(self->m_poll_mutex);

    // Initialize the hint value for lookups.
    size_t index = self->m_poll_elems.size();
    if(index > poll_index_max)
      POSEIDON_THROW("too many simultaneous connections");

    // Make sure later `emplace_back()` will not throw an exception.
    self->m_poll_elems.reserve(index + 1);

    // Add the socket for polling.
    ::epoll_event event;
    event.data.u64 = self->make_epoll_data(index, self->m_poll_serial++);
    event.events = EPOLLIN | EPOLLOUT | EPOLLET;
    if(::epoll_ctl(self->m_epoll_fd, EPOLL_CTL_ADD, sock->get_fd(), &event) != 0)
      POSEIDON_THROW(
          "failed to add socket into epoll\n"
          "[`epoll_ctl()` failed: $1]",
          format_errno());

    // Initialize epoll data.
    sock->m_epoll_data = event.data.u64;
    sock->m_epoll_events = 0;

    // Push the new element.
    // Space has been reserved so no exception can be thrown.
    Poll_Socket elem;
    elem.sock = sock;
    self->m_poll_elems.emplace_back(::std::move(elem));
    POSEIDON_LOG_TRACE("Socket added: $1", sock);
    return sock;
  }

bool
Network_Driver::
notify_writable_internal(const Abstract_Socket& sock) noexcept
  {
    // If the socket has been removed or is not writable , don't do anything.
    simple_mutex::unique_lock lock(self->m_poll_mutex);
    if((sock.m_epoll_events & (EPOLLOUT | EPOLLERR | EPOLLHUP)) != EPOLLOUT)
      return false;

    // Don't do anything if the socket does not exist in epoll.
    uint32_t index = self->find_poll_socket(sock.m_epoll_data);
    if(index == poll_index_nil)
      return false;

    // Append the socket to write list if writing is possible.
    // If the network thread might be blocking on epoll, wake it up.
    self->do_signal_if_poll_lists_empty();
    self->poll_list_attach(self->m_poll_root_wr, index);
    return true;
  }

}  // namespace poseidon
