// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_LOG_HPP_
#define POSEIDON_LOG_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include <sstream>
#include <cstddef>

namespace Poseidon {

class Logger : NONCOPYABLE {
public:
	enum {
		LV_FATAL		= 0x0001,
		LV_ERROR		= 0x0002,
		LV_WARN			= 0x0004,
		LV_INFO			= 0x0008,
		LV_DEBUG		= 0x0010,
		LV_TRACE		= 0x0020,

		SRC_RESERVED	= 0x0040,
		SRC_POSEIDON	= 0x0080,
	};

public:
	static unsigned long long getMask() NOEXCEPT;
	static unsigned long long setMask(
		unsigned long long toDisable, unsigned long long toEnable) NOEXCEPT;

	static const char *getThreadTag() NOEXCEPT;
	static void setThreadTag(const char *newTag) NOEXCEPT;

private:
	const unsigned long long m_mask;
	const char *const m_file;
	const std::size_t m_line;

	std::stringstream m_stream;

public:
	Logger(unsigned long long mask, const char *file, std::size_t line) NOEXCEPT;
	~Logger() NOEXCEPT;

private:
	template<typename T>
	void put(const T &val){
		m_stream <<val;
	}

	void put(bool val){
		m_stream <<std::boolalpha <<val;
	}
	void put(signed char val){
		m_stream <<(int)val;
	}
	void put(unsigned char val){
		m_stream <<(unsigned)val;
	}
	void put(const volatile signed char *val){
		m_stream <<(const void *)val;
	}
	void put(const volatile unsigned char *val){
		m_stream <<(const void *)val;
	}

public:
	template<typename T>
	Logger &operator,(const T &val) NOEXCEPT {
		try {
			put(val);
		} catch(...){
		}
		return *this;
	}
};

}

#define LOG_MASK(mask_, ...)	\
	do {	\
		if(((mask_) & ~::Poseidon::Logger::getMask()) == 0){	\
			static_cast<void>(::Poseidon::Logger(mask_, __FILE__, __LINE__), __VA_ARGS__);	\
		}	\
	} while(false)

#define LOG_POSEIDON(level_, ...)	\
	LOG_MASK(::Poseidon::Logger::SRC_POSEIDON | (level_), __VA_ARGS__)

#define LOG_POSEIDON_FATAL(...)		LOG_POSEIDON(::Poseidon::Logger::LV_FATAL,	__VA_ARGS__)
#define LOG_POSEIDON_ERROR(...)		LOG_POSEIDON(::Poseidon::Logger::LV_ERROR,	__VA_ARGS__)
#define LOG_POSEIDON_WARN(...)		LOG_POSEIDON(::Poseidon::Logger::LV_WARN,	__VA_ARGS__)
#define LOG_POSEIDON_INFO(...)		LOG_POSEIDON(::Poseidon::Logger::LV_INFO,	__VA_ARGS__)
#define LOG_POSEIDON_DEBUG(...)		LOG_POSEIDON(::Poseidon::Logger::LV_DEBUG,	__VA_ARGS__)
#define LOG_POSEIDON_TRACE(...)		LOG_POSEIDON(::Poseidon::Logger::LV_TRACE,	__VA_ARGS__)

#endif
