#ifndef POSEIDON_LOG_HPP_
#define POSEIDON_LOG_HPP_

#include <sstream>
#include <cstddef>
#include <boost/noncopyable.hpp>

namespace Poseidon {

class Log : private boost::noncopyable {
public:
	enum {
		FATAL,
		ERROR,
		WARNING,
		INFO,
		DEBUG,
	};

public:
	static unsigned getLogLevel();
	static void setLogLevel(unsigned newLevel);

private:
	const char *const m_comment;
	const char *const m_file;
	const std::size_t m_line;

	std::stringstream m_stream;

public:
	Log(const char *comment, const char *file, std::size_t line) throw();
	~Log() throw();

public:
	template<typename T>
	Log &operator<<(const T &info) throw() {
		try {
			m_stream <<info;
		} catch(...){
		}
		return *this;
	}
};

}

#define LOG_LEVEL(level)	\
	if(::Poseidon::Log::getLogLevel() >= ::Poseidon::Log::level)	\
		::Poseidon::Log(#level, __FILE__, __LINE__)

#define LOG_FATAL		LOG_LEVEL(FATAL)
#define LOG_ERROR		LOG_LEVEL(ERROR)
#define LOG_WARNING		LOG_LEVEL(WARNING)
#define LOG_INFO		LOG_LEVEL(INFO)
#define LOG_DEBUG		LOG_LEVEL(DEBUG)

#endif
