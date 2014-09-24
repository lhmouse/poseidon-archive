#ifndef POSEIDON_LOG_HPP_
#define POSEIDON_LOG_HPP_

#include "../cxx_ver.hpp"
#include <sstream>
#include <cstddef>
#include <boost/noncopyable.hpp>

namespace Poseidon {

class Log : boost::noncopyable {
public:
	enum {
		LV_FATAL,		// 0
		LV_ERROR,		// 1
		LV_WARNING,		// 2
		LV_INFO,		// 3
		LV_DEBUG,		// 4
	};

	enum {
		TAG_PRIMARY,	// 0
		TAG_TIMER,		// 1
		TAG_DATABASE,	// 2
		TAG_EPOLL,		// 3
	};

public:
	static unsigned getLevel();
	static void setLevel(unsigned newLevel);

	static unsigned getThreadTag();
	static void setThreadTag(unsigned newTag);

private:
	const unsigned m_level;
	const char *const m_comment;
	const char *const m_file;
	const std::size_t m_line;

	std::stringstream m_stream;

public:
	Log(unsigned level, const char *comment,
		const char *file, std::size_t line) NOEXCEPT;
	~Log() NOEXCEPT;

public:
	template<typename T>
	Log &operator,(const T &info) NOEXCEPT {
		try {
			m_stream <<info;
		} catch(...){
		}
		return *this;
	}

	Log &operator,(signed char ch) NOEXCEPT {
		try {
			m_stream <<(signed)ch;
		} catch(...){
		}
		return *this;
	}
	Log &operator,(unsigned char ch) NOEXCEPT {
		try {
			m_stream <<(unsigned)ch;
		} catch(...){
		}
		return *this;
	}

	Log &operator,(const signed char *p) NOEXCEPT {
		try {
			m_stream <<(const void *)p;
		} catch(...){
		}
		return *this;
	}
	Log &operator,(const unsigned char *p) NOEXCEPT {
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
		if((long)::Poseidon::Log::getLevel() + 1 >=	\
			(long)::Poseidon::Log::LV_ ## level_ + 1)	\
		{	\
			::Poseidon::Log(::Poseidon::Log::LV_ ## level_,	\
			 	#level_, __FILE__, __LINE__), __VA_ARGS__;	\
		}	\
	} while(false)

#define LOG_FATAL(...)		LOG_LEVEL(FATAL, __VA_ARGS__)
#define LOG_ERROR(...)		LOG_LEVEL(ERROR, __VA_ARGS__)
#define LOG_WARNING(...)	LOG_LEVEL(WARNING, __VA_ARGS__)
#define LOG_INFO(...)		LOG_LEVEL(INFO, __VA_ARGS__)
#define LOG_DEBUG(...)		LOG_LEVEL(DEBUG, __VA_ARGS__)

#endif
