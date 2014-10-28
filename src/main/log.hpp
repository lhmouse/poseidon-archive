#ifndef POSEIDON_LOG_HPP_
#define POSEIDON_LOG_HPP_

#include "cxx_ver.hpp"
#include <sstream>
#include <cstddef>
#include <boost/noncopyable.hpp>

namespace Poseidon {

class Logger : boost::noncopyable {
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

public:
	template<typename T>
	Logger &operator,(const T &info) NOEXCEPT {
		try {
			m_stream <<info;
		} catch(...){
		}
		return *this;
	}

	Logger &operator,(signed char ch) NOEXCEPT {
		try {
			m_stream <<(signed)ch;
		} catch(...){
		}
		return *this;
	}
	Logger &operator,(unsigned char ch) NOEXCEPT {
		try {
			m_stream <<(unsigned)ch;
		} catch(...){
		}
		return *this;
	}

	Logger &operator,(const signed char *p) NOEXCEPT {
		try {
			m_stream <<(const void *)p;
		} catch(...){
		}
		return *this;
	}
	Logger &operator,(const unsigned char *p) NOEXCEPT {
		try {
			m_stream <<(const void *)p;
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

#define LOG_FATAL(...)		LOG_POSEIDON(::Poseidon::Logger::LV_FATAL,	__VA_ARGS__)
#define LOG_ERROR(...)		LOG_POSEIDON(::Poseidon::Logger::LV_ERROR,	__VA_ARGS__)
#define LOG_WARN(...)		LOG_POSEIDON(::Poseidon::Logger::LV_WARN,	__VA_ARGS__)
#define LOG_INFO(...)		LOG_POSEIDON(::Poseidon::Logger::LV_INFO,	__VA_ARGS__)
#define LOG_DEBUG(...)		LOG_POSEIDON(::Poseidon::Logger::LV_DEBUG,	__VA_ARGS__)
#define LOG_TRACE(...)		LOG_POSEIDON(::Poseidon::Logger::LV_TRACE,	__VA_ARGS__)

#endif
