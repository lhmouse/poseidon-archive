// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "network_driver.hpp"
#include "main_config.hpp"
#include "../core/config_file.hpp"
#include "../socket/abstract_socket.hpp"
#include "../xutilities.hpp"
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/eventfd.h>

namespace poseidon {
namespace {

size_t
do_get_size_config(const Config_File& file, const char* name, size_t defval)
  {
    const auto qval = file.get_int64_opt({"network","poll",name});
    if(!qval)
      return defval;

    int64_t rval = ::rocket::clamp(*qval, 1, 0x10'00000);   // 16MiB
    if(*qval != rval)
      POSEIDON_LOG_WARN("Config value `network.poll.$1` truncated to `$2`\n"
                        "[value `$3` out of range]",
                        name, rval, *qval);

    return static_cast<size_t>(rval);
  }

struct Config_Scalars
  {
    size_t event_buffer_size = 1;
    size_t io_buffer_size = 1;
    size_t throttle_size = 1;
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

void
do_event_wait(int fd)
noexcept
  {
    int err;

    // Repeat reading until the operation fails with something other than EINTR.
    for(;;) {
      ::eventfd_t ignored;
      if(::eventfd_read(fd, &ignored) == 0)
        continue;

      err = errno;
      if(err != EINTR)
        break;
    }

    if(::rocket::is_none_of(err, { EAGAIN, EWOULDBLOCK }))
      POSEIDON_LOG_FATAL("`eventfd_read()` failed: $1", noadl::format_errno(err));
  }

void
do_event_signal(int fd)
noexcept
  {
    int err;

    // Repeat writing until the operation succeeds or fails with something other than EINTR.
    for(;;) {
      err = 0;
      if(::eventfd_write(fd, 1) == 0)
        break;

      err = errno;
      if(err != EINTR)
        break;
    }

    if(err != 0)
      POSEIDON_LOG_FATAL("`eventfd_write()` failed: $1", noadl::format_errno(err));
  }

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Network_Driver)
  {
    // constant data
    ::rocket::once_flag m_init_once;
    ::pthread_t m_thread;
    int m_epoll_fd = -1;
    int m_event_fd = -1;

    // configuration
    mutable mutex m_conf_mutex;
    Config_Scalars m_conf;

    // dynamic data
    ::std::vector<::epoll_event> m_event_buffer;
    ::std::vector<rcptr<Abstract_Socket>> m_ready_socks;
    ::std::vector<uint8_t> m_io_buffer;

    template<Poll_List_mixin Poll_Socket::* mptrT>
    struct Poll_List_root
      {
        uint32_t head = poll_index_end;
        uint32_t tail = poll_index_end;
      };

    mutable mutex m_poll_mutex;
    ::std::vector<Poll_Socket> m_poll_elems;
    uint64_t m_poll_serial = 0;
    Poll_List_root<&Poll_Socket::node_cl> m_poll_root_cl;
    Poll_List_root<&Poll_Socket::node_rd> m_poll_root_rd;
    Poll_List_root<&Poll_Socket::node_wr> m_poll_root_wr;

    static
    void
    do_init_once()
      {
        // Create an epoll object.
        unique_FD epoll_fd(::epoll_create(100));
        if(!epoll_fd)
          POSEIDON_THROW("Could not create epoll object\n"
                         "[`epoll_create()` failed: $1]",
                         format_errno(errno));

        // Create the notification eventfd and add it into epoll.
        unique_FD event_fd(::eventfd(0, EFD_NONBLOCK));
        if(!event_fd)
          POSEIDON_THROW("Could not create eventfd object\n"
                         "[`eventfd()` failed: $1]",
                         format_errno(errno));

        ::epoll_event event;
        event.data.u64 = self->make_epoll_data(poll_index_event, 0);
        event.events = EPOLLIN | EPOLLET;
        if(::epoll_ctl(epoll_fd, EPOLL_CTL_ADD, event_fd, &event) != 0)
          POSEIDON_THROW("Failed to add socket into epoll\n"
                         "[`epoll_ctl()` failed: $1]",
                         noadl::format_errno(errno));

        // Create the thread. Note it is never joined or detached.
        mutex::unique_lock lock(self->m_conf_mutex);
        self->m_thread = create_daemon_thread<do_thread_loop>("network");
        self->m_epoll_fd = epoll_fd.release();
        self->m_event_fd = event_fd.release();
      }

    // The index takes up 24 bits. That's 16M simultaneous connections.
    // The serial number takes up 40 bits. That's ~1.1T historical connections.
    static constexpr
    uint64_t
    make_epoll_data(uint64_t index, uint64_t serial)
    noexcept
      {
        return (index << 40) | ((serial << 24) >> 24);
      }

    static constexpr
    uint32_t
    index_from_epoll_data(uint64_t epoll_data)
    noexcept
      {
        return static_cast<uint32_t>(epoll_data >> 40);
      }

    // Note epoll events are bound to kernel files, not individual file descriptors.
    // If we were passing pointers in `event.data` and FDs got `dup()`'d, we could get
    // dangling pointers here, which is rather dangerous.
    static
    uint32_t
    find_poll_socket(uint64_t epoll_data)
    noexcept
      {
        // Perform fast lookup using the hint value.
        uint32_t index = self->index_from_epoll_data(epoll_data);
        if(ROCKET_EXPECT((index < self->m_poll_elems.size())
                         && (self->m_poll_elems[index].sock->m_epoll_data == epoll_data))) {
          // Hint valid.
          ROCKET_ASSERT(index != poll_index_nil);
          return index;
        }
        POSEIDON_LOG_DEBUG("Epoll lookup hint invalidated: value = $1", epoll_data);

        // Perform a brute-force search.
        for(index = 0;  index < self->m_poll_elems.size();  ++index) {
          const auto& elem = self->m_poll_elems[index];
          if(elem.sock->m_epoll_data != epoll_data)
            continue;

          // Update lookup hint. Errors are ignored.
          ::epoll_event event;
          event.data.u64 = self->make_epoll_data(index, epoll_data);
          event.events = EPOLLIN | EPOLLOUT | EPOLLET;
          if(::epoll_ctl(self->m_epoll_fd, EPOLL_CTL_MOD, elem.sock->get_fd(), &event) != 0)
            POSEIDON_LOG_FATAL("failed to modify socket in epoll\n"
                               "[`epoll_ctl()` failed: $1]",
                               noadl::format_errno(errno));

          elem.sock->m_epoll_data = event.data.u64;
          POSEIDON_LOG_DEBUG("Epoll lookup hint updated: value = $1", event.data.u64);

          // Return the new index.
          ROCKET_ASSERT(index != poll_index_nil);
          return index;
        }

        POSEIDON_LOG_ERROR("Socket not found: epoll_data = $1", epoll_data);
        return poll_index_nil;
      }

    ROCKET_PURE_FUNCTION static
    bool
    poll_lists_empty()
    noexcept
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
    poll_list_attach(Poll_List_root<mptrT>& root, uint32_t index)
    noexcept
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
    poll_list_detach(Poll_List_root<mptrT>& root, uint32_t index)
    noexcept
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
    size_t
    do_epoll_wait()
    noexcept
      {
        int ret = ::epoll_wait(self->m_epoll_fd, self->m_event_buffer.data(),
                                                 static_cast<int>(self->m_event_buffer.size()),
                                                 60'000);  // one minute
        if(ROCKET_EXPECT(ret >= 0))
          return static_cast<uint32_t>(ret);

        int err = errno;
        if(err != EINTR)
          POSEIDON_LOG_FATAL("`epoll_wait()` failed: $1", noadl::format_errno(err));

        return 0;  // no events
      }

    static
    void
    do_thread_loop(void* /*param*/)
      {
        // Reload configuration.
        mutex::unique_lock lock(self->m_conf_mutex);
        const auto conf = self->m_conf;
        lock.unlock();

        self->m_event_buffer.resize(conf.event_buffer_size);
        self->m_ready_socks.clear();
        self->m_io_buffer.resize(conf.io_buffer_size);

        // Try polling if there is nothing to do.
        lock.assign(self->m_poll_mutex);
        if(self->poll_lists_empty()) {
          // Delete sockets that have no other references to them.
          for(const auto& elem : self->m_poll_elems) {
            bool kill = elem.sock.unique() && !elem.sock->resident();
            if(!kill)
              continue;

            POSEIDON_LOG_DEBUG("Killed orphan socket: $1", elem.sock);
            elem.sock->abort();
          }
          lock.unlock();

          // Await I/O events.
          size_t nevents = self->do_epoll_wait();

          // Process all events that have been received so far.
          // Note the loop below will not throw exceptions.
          lock.assign(self->m_poll_mutex);
          for(size_t k = 0;  k < nevents;  ++k) {
            ::epoll_event& event = self->m_event_buffer[k];

            // The special event is not associated with any socket.
            if(self->index_from_epoll_data(event.data.u64) == poll_index_event) {
              do_event_wait(self->m_event_fd);
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
        lock.assign(self->m_poll_mutex);
        self->poll_list_collect(self->m_poll_root_cl);
        for(const auto& sock : self->m_ready_socks) {
          // Set `err` to zero if normal closure, and to the error code otherwise.
          lock.assign(self->m_poll_mutex);
          int err = sock->m_epoll_events & EPOLLERR;
          lock.unlock();

          if(err) {
            ::socklen_t optlen = sizeof(err);
            if(::getsockopt(sock->get_fd(), SOL_SOCKET, SO_ERROR, &err, &optlen) != 0)
              err = errno;
          }
          POSEIDON_LOG_TRACE("Socket closed: $1 ($2)", sock, noadl::format_errno(err));

          // Remove the socket from epoll. Errors are ignored.
          if(::epoll_ctl(self->m_epoll_fd, EPOLL_CTL_DEL, sock->get_fd(), (::epoll_event*)1) != 0)
            POSEIDON_LOG_FATAL("failed to remove socket from epoll\n"
                               "[`epoll_ctl()` failed: $1]",
                               noadl::format_errno(errno));

          try {
            sock->do_on_async_poll_shutdown(err);
          }
          catch(exception& stdex) {
            POSEIDON_LOG_WARN("Socket shutdown error: $1\n"
                              "[socket class `$2`]",
                              stdex.what(), typeid(*sock).name());
          }

          // Remove the socket, no matter whether an exception was thrown or not.
          lock.assign(self->m_poll_mutex);
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
          POSEIDON_LOG_TRACE("Socket removed: $1", sock);
        }

        // Process readable sockets.
        lock.assign(self->m_poll_mutex);
        self->poll_list_collect(self->m_poll_root_rd);
        for(const auto& sock : self->m_ready_socks) {
          // Perform a single read operation (no retry upon EINTR).
          lock.unlock();

          bool throttle;
          bool detach;
          bool clear_status;

          try {
            // Check whether the socket should be throttled.
            throttle = sock->do_write_queue_size(lock) > conf.throttle_size;
            if(throttle) {
              // If the socket is throttled, remove it from read queue.
              detach = true;
              clear_status = false;
            }
            else {
              // If the socket is not throttled, try reading some bytes.
              auto io_res = sock->do_on_async_poll_read(lock,
                                       self->m_io_buffer.data(), self->m_io_buffer.size());

              // If the read operation reports `io_result_again` or `io_result_eof`, the socket
              // shall be removed from read queue and the `EPOLLIN` status shall be cleared.
              detach = io_res <= 0;
              clear_status = io_res <= 0;
            }
          }
          catch(exception& stdex) {
            POSEIDON_LOG_WARN("Socket read error: $1\n"
                              "[socket class `$2`]",
                              stdex.what(), typeid(*sock).name());

            // Close the connection.
            sock->abort();

            // If a read error occurs, the socket shall be removed from read queue and the
            // `EPOLLIN` status shall be cleared.
            throttle = false;
            detach = true;
            clear_status = true;
          }

          // Update the socket.
          lock.assign(self->m_poll_mutex);
          uint32_t index = self->find_poll_socket(sock->m_epoll_data);
          if(index == poll_index_nil)
            continue;

          if(detach)
            self->poll_list_detach(self->m_poll_root_rd, index);

          if(clear_status)
            sock->m_epoll_events &= ~EPOLLIN;
        }

        // Process writable sockets.
        lock.assign(self->m_poll_mutex);
        self->poll_list_collect(self->m_poll_root_wr);
        for(const auto& sock : self->m_ready_socks) {
          // Perform a single write operation (no retry upon EINTR).
          lock.unlock();

          bool detach;
          bool clear_status;
          bool throttle;

          try {
            // Try writing some bytes.
            auto io_res = sock->do_on_async_poll_write(lock,
                                       self->m_io_buffer.data(), self->m_io_buffer.size());

            // If the write operation reports `io_result_again` or `io_result_eof`, the socket
            // shall be removed from write queue.
            detach = io_res <= 0;

            // If the write operation reports `io_result_again`, in addition to the removal,
            // the `EPOLLOUT` status shall be cleared.
            clear_status = io_res == io_result_again;

            // Check whether the socket should be unthrottled.
            throttle = sock->do_write_queue_size(lock) > conf.throttle_size;
          }
          catch(exception& stdex) {
            POSEIDON_LOG_WARN("Socket write error: $1\n"
                              "[socket class `$2`]",
                              stdex.what(), typeid(*sock).name());

            // Close the connection.
            sock->abort();

            // If a write error occurs, the socket shall be removed from write queue and the
            // `EPOLLOUT` status shall be cleared.
            detach = true;
            clear_status = true;
            throttle = true;
          }

          // Update the socket.
          lock.assign(self->m_poll_mutex);
          uint32_t index = self->find_poll_socket(sock->m_epoll_data);
          if(index == poll_index_nil)
            continue;

          if(detach)
            self->poll_list_detach(self->m_poll_root_wr, index);

          if(clear_status)
            sock->m_epoll_events &= ~EPOLLOUT;

          if(!throttle && (sock->m_epoll_events & EPOLLIN))
            self->poll_list_attach(self->m_poll_root_rd, index);
        }
      }

    static
    void
    do_signal_if_poll_lists_empty()
    noexcept
      {
        if(ROCKET_EXPECT(!self->poll_lists_empty()))
          return;

        do_event_signal(self->m_event_fd);
      }
  };

void
Network_Driver::
start()
  {
    self->m_init_once.call(self->do_init_once);
  }

void
Network_Driver::
reload()
  {
    // Load logger settings into temporary objects.
    auto file = Main_Config::copy();

    Config_Scalars conf;
    conf.event_buffer_size = do_get_size_config(file, "event_buffer_size", 1024);
    conf.io_buffer_size = do_get_size_config(file, "io_buffer_size", 65536);
    conf.throttle_size = do_get_size_config(file, "throttle_size", 1048576);

    // During destruction of temporary objects the mutex should have been unlocked.
    // The swap operation is presumed to be fast, so we don't hold the mutex
    // for too long.
    mutex::unique_lock lock(self->m_conf_mutex);
    self->m_conf = conf;
  }

rcptr<Abstract_Socket>
Network_Driver::
insert(uptr<Abstract_Socket>&& usock)
  {
    // Take ownership of `usock`.
    rcptr<Abstract_Socket> sock(usock.release());
    if(!sock)
      POSEIDON_THROW("Null socket pointer not valid");

    if(!sock.unique())
      POSEIDON_THROW("Socket pointer must be unique");

    // Lock epoll for modification.
    mutex::unique_lock lock(self->m_poll_mutex);

    // Initialize the hint value for lookups.
    size_t index = self->m_poll_elems.size();
    if(index > poll_index_max)
      POSEIDON_THROW("Too many simultaneous connections");

    // Make sure later `emplace_back()` will not throw an exception.
    self->m_poll_elems.reserve(index + 1);

    // Add the socket for polling.
    ::epoll_event event;
    event.data.u64 = self->make_epoll_data(index, self->m_poll_serial++);
    event.events = EPOLLIN | EPOLLOUT | EPOLLET;
    if(::epoll_ctl(self->m_epoll_fd, EPOLL_CTL_ADD, sock->get_fd(), &event) != 0)
      POSEIDON_THROW("Failed to add socket into epoll\n"
                     "[`epoll_ctl()` failed: $1]",
                     noadl::format_errno(errno));

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
notify_writable_internal(const Abstract_Socket* csock)
noexcept
  {
    // If the socket has been removed or is not writable , don't do anything.
    mutex::unique_lock lock(self->m_poll_mutex);
    if((csock->m_epoll_events & (EPOLLOUT | EPOLLERR | EPOLLHUP)) != EPOLLOUT)
      return false;

    // Don't do anything if the socket does not exist in epoll.
    uint32_t index = self->find_poll_socket(csock->m_epoll_data);
    if(index == poll_index_nil)
      return false;

    // Append the socket to write list if writing is possible.
    // If the network thread might be blocking on epoll, wake it up.
    self->do_signal_if_poll_lists_empty();
    self->poll_list_attach(self->m_poll_root_wr, index);
    return true;
  }

}  // namespace poseidon
