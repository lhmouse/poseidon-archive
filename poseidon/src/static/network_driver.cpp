// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "network_driver.hpp"
#include "main_config.hpp"
#include "../core/config_file.hpp"
#include "../network/abstract_socket.hpp"
#include "../xutilities.hpp"
#include <sys/epoll.h>

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
                        "[value `$3` was out of range]",
                        name, rval, *qval);

    return static_cast<size_t>(rval);
  }

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Network_Driver)
  {
    ::pthread_t m_thread;
    int m_epoll = -1;

    mutable Si_Mutex m_conf_mutex;
    size_t m_conf_event_count = 1;
    size_t m_conf_io_buffer_size = 1;
    size_t m_conf_throttle_size = 1;

    mutable Si_Mutex m_poll_mutex;
    
  };

void
Network_Driver::
do_thread_loop(void* /*param*/)
  {
    POSEIDON_LOG_FATAL("EPOLL RUNNING");
    sleep(1);
  }

void
Network_Driver::
start()
  {
    if(self->m_thread)
      return;

    // Create an epoll object.
    if(self->m_epoll == -1) {
      int fd = ::epoll_create(100);
      if(fd == -1)
        POSEIDON_THROW("could not create epoll object\n"
                       "[`epoll_create()` failed: $1]",
                       format_errno(errno));

      self->m_epoll = fd;
    }

    // Create the thread. Note it is never joined or detached.
    auto thr = create_daemon_thread<do_thread_loop>("network");
    self->m_thread = ::std::move(thr);
  }

void
Network_Driver::
reload()
  {
    // Load logger settings into temporary objects.
    auto file = Main_Config::copy();
    size_t event_count = do_get_size_config(file, "event_count", 1024);
    size_t io_buffer_size = do_get_size_config(file, "io_buffer_size", 65536);
    size_t throttle_size = do_get_size_config(file, "throttle_size", 1048576);

    // During destruction of temporary objects the mutex should have been unlocked.
    // The swap operation is presumed to be fast, so we don't hold the mutex
    // for too long.
    Si_Mutex::unique_lock lock(self->m_conf_mutex);
    self->m_conf_event_count = event_count;
    self->m_conf_io_buffer_size = io_buffer_size;
    self->m_conf_throttle_size = throttle_size;
  }

rcptr<Abstract_Socket>
Network_Driver::
insert(uptr<Abstract_Socket>&& usock)
  {
    return nullptr;
  }

bool
Network_Driver::
notify_writable_internal(const Abstract_Socket* csock)
noexcept
  {
    return 1;
  }

}  // namespace poseidon
