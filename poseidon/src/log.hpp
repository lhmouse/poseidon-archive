// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_LOG_HPP_
#define POSEIDON_LOG_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "buffer_streams.hpp"
#include <cstddef>
#include <boost/cstdint.hpp>

namespace Poseidon {

class Logger : NONCOPYABLE {
public:
	enum {
		special_poseidon = 0x80,
		special_major    = 0x40,
		level_trace      = 0x20,
		level_debug      = 0x10,
		level_info       = 0x08,
		level_warning    = 0x44,
		level_error      = 0x42,
		level_fatal      = 0x41,
	};

public:
	static boost::uint64_t get_mask() NOEXCEPT;
	static boost::uint64_t set_mask(boost::uint64_t to_disable, boost::uint64_t to_enable) NOEXCEPT;

	static bool check_mask(boost::uint64_t mask) NOEXCEPT {
		return (mask & ((mask & special_major) - 1) & ~get_mask()) == 0;
	}

	static bool initialize_mask_from_config();
	static void finalize_mask() NOEXCEPT;

	static const char * get_thread_tag() NOEXCEPT;
	static void set_thread_tag(const char *tag) NOEXCEPT;

private:
	const boost::uint64_t m_mask;
	const char *const m_file;
	const std::size_t m_line;

	Buffer_ostream m_stream;

public:
	Logger(boost::uint64_t mask, const char *file, std::size_t line) NOEXCEPT;
	~Logger() NOEXCEPT;

private:
	// operator<< 的 name lookup 拖慢编译速度。
	void put(bool val);
	void put(char val);
	void put(signed char val);
	void put(unsigned char val);
	void put(short val);
	void put(unsigned short val);
	void put(int val);
	void put(unsigned val);
	void put(long val);
	void put(unsigned long val);
	void put(long long val);
	void put(unsigned long long val);
	void put(const char *val);
	void put(const signed char *val);
	void put(const unsigned char *val);
	void put(const void *val);

	template<typename T>
	void put(const T &val){
		m_stream <<val;
	}

public:
	template<typename T>
	Logger & operator,(const T &val) NOEXCEPT
	try {
		this->put(val);
		return *this;
	} catch(...){
		return *this;
	}
};

}

#define POSEIDON_CHECK_AND_LOG(mask_, ...)  	(::Poseidon::Logger::check_mask(mask_) && (static_cast<void>(::Poseidon::Logger(mask_, __FILE__, __LINE__), __VA_ARGS__), 1))

#define POSEIDON_LOG(lv_, ...)      POSEIDON_CHECK_AND_LOG((lv_) | ::Poseidon::Logger::special_poseidon, __VA_ARGS__)
#define POSEIDON_LOG_FATAL(...)     POSEIDON_LOG(::Poseidon::Logger::level_fatal,   __VA_ARGS__)
#define POSEIDON_LOG_ERROR(...)     POSEIDON_LOG(::Poseidon::Logger::level_error,   __VA_ARGS__)
#define POSEIDON_LOG_WARNING(...)   POSEIDON_LOG(::Poseidon::Logger::level_warning, __VA_ARGS__)
#define POSEIDON_LOG_INFO(...)      POSEIDON_LOG(::Poseidon::Logger::level_info,    __VA_ARGS__)
#define POSEIDON_LOG_DEBUG(...)     POSEIDON_LOG(::Poseidon::Logger::level_debug,   __VA_ARGS__)
#define POSEIDON_LOG_TRACE(...)     POSEIDON_LOG(::Poseidon::Logger::level_trace,   __VA_ARGS__)

#endif
