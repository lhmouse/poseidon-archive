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
		LV_DISABLED,	// 0
		LV_FATAL,		// 1
		LV_ERROR,		// 2
		LV_WARN,		// 3
		LV_INFO,		// 4
		LV_DEBUG,		// 5
		LV_TRACE,		// 6
	};

public:
	static unsigned getLevel();
	static unsigned setLevel(unsigned newLevel);

	static const char *getThreadTag();
	static void setThreadTag(const char *newTag);

private:
	const unsigned m_level;
	const char *const m_file;
	const std::size_t m_line;

	std::stringstream m_stream;

public:
	Logger(unsigned level, const char *file, std::size_t line) NOEXCEPT;
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

#define LOG_LEVEL(level_, ...)	\
	do {	\
		if(level_ <= ::Poseidon::Logger::getLevel()){	\
			static_cast<void>(::Poseidon::Logger(level_, __FILE__, __LINE__), __VA_ARGS__);	\
		}	\
	} while(false)

#define LOG_FATAL(...)		LOG_LEVEL(::Poseidon::Logger::LV_FATAL, __VA_ARGS__)
#define LOG_ERROR(...)		LOG_LEVEL(::Poseidon::Logger::LV_ERROR, __VA_ARGS__)
#define LOG_WARN(...)		LOG_LEVEL(::Poseidon::Logger::LV_WARN, __VA_ARGS__)
#define LOG_INFO(...)		LOG_LEVEL(::Poseidon::Logger::LV_INFO, __VA_ARGS__)
#define LOG_DEBUG(...)		LOG_LEVEL(::Poseidon::Logger::LV_DEBUG, __VA_ARGS__)
#define LOG_TRACE(...)		LOG_LEVEL(::Poseidon::Logger::LV_TRACE, __VA_ARGS__)

#endif
