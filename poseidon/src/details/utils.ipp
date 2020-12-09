// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UTILS_HPP_
#  error Please include <poseidon/utils.hpp> instead.
#endif

namespace poseidon {
namespace details_utils {

template<typename... ParamsT>
ROCKET_NOINLINE inline
void
format_log(Log_Level level, const char* file, long line, const char* func,
           const char* templ, const ParamsT&... params)
  noexcept
  try {
    // Compose the message.
    ::rocket::tinyfmt_str fmt;
    format(fmt, templ, params...);  // ADL intended
    auto text = fmt.extract_string();

    // Push a new log entry.
    Async_Logger::enqueue(level, file, line, func, ::std::move(text));
  }
  catch(exception& stdex) {
    // Ignore this exception, but print a message.
    ::std::fprintf(stderr,
        "%s: %s\n"
        "[exception class `%s` thrown from '%s:%ld']\n",
        func, stdex.what(),
        typeid(stdex).name(), file, line);
  }

template<typename... ParamsT>
[[noreturn]] ROCKET_NOINLINE inline
void
format_throw(const char* file, long line, const char* func,
             const char* templ, const ParamsT&... params)
  {
    // Compose the message.
    ::rocket::tinyfmt_str fmt;
    format(fmt, templ, params...);  // ADL intended
    auto text = fmt.extract_string();

    // Push a new log entry.
    static constexpr auto level = log_level_warn;
    if(Async_Logger::enabled(level))
      Async_Logger::enqueue(level, file, line, func, "POSEIDON_THROW: " + text);

    // Throw the exception.
    ::rocket::sprintf_and_throw<::std::runtime_error>(
        "%s: %s\n"
        "[thrown from '%s:%ld']",
        func, text.c_str(),
        file, line);
  }

template<void loopfnT(void*)>
[[noreturn]] ROCKET_NOINLINE inline
void*
daemon_thread_proc(void* param)
  {
    // Disable cancellation for safety.
    // Failure to set the cancel state is ignored.
    int state;
    ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &state);

    // Execute `loopfnT` repeatedly.
    // The thread never exits.
    do
      try {
        loopfnT(param);
      }
      catch(exception& stdex) {
        ::std::fprintf(stderr,
            "%s: %s\n"
            "[exception class `%s` thrown from daemon thread loop]\n",
            __func__, stdex.what(),
            typeid(stdex).name());
      }
    while(true);
  }

template<typename FuncT>
class Timer
  final
  : public Abstract_Timer
  {
  private:
    FuncT m_func;

  public:
    template<typename... ParamsT>
    explicit
    Timer(int64_t next, int64_t period, ParamsT&&... params)
      : Abstract_Timer(next, period),
        m_func(::std::forward<ParamsT>(params)...)
      { }

  private:
    void
    do_on_async_timer(int64_t now)
      override
      { this->m_func(now);  }

  public:
    ~Timer()
      override;
  };

template<typename FuncT>
Timer<FuncT>::
~Timer()
  = default;

struct random_key_t
  { }
  constexpr random_key;

template<typename FuncT>
class Async
  final
  : public Abstract_Async_Job
  {
  public:
    using result_type = typename ::std::result_of<FuncT& ()>::type;

  private:
    prom<result_type> m_prom;
    FuncT m_func;

  public:
    template<typename... ParamsT>
    explicit
    Async(uintptr_t key, ParamsT&&... params)
      : Abstract_Async_Job(key),
        m_func(::std::forward<ParamsT>(params)...)
      { }

    template<typename... ParamsT>
    explicit
    Async(random_key_t, ParamsT&&... params)
      : Abstract_Async_Job(reinterpret_cast<uintptr_t>(this) / alignof(max_align_t)),
        m_func(::std::forward<ParamsT>(params)...)
      { }

  private:
    void
    do_execute()
      override
      try {
        this->m_prom.set_value(this->m_func());
      }
      catch(...) {
        this->m_prom.set_exception(::std::current_exception());
      }

  public:
    ~Async()
      override;

    futp<result_type>
    future()
      const
      { return this->m_prom.future();  }
  };

template<typename FuncT>
Async<FuncT>::
~Async()
  = default;

template<typename AsyncT>
inline
futp<typename AsyncT::result_type>
promise(uptr<AsyncT>&& async)
  {
    auto futr = async->future();
    Worker_Pool::insert(::std::move(async));
    return futr;
  }

}  // namespace details_utils
}  // namespace poseidon
