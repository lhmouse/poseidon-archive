// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "log.hpp"
#include "atomic.hpp"
#include "time.hpp"
#include "flags.hpp"
#include "singletons/main_config.hpp"
#include <unistd.h>
#include <sys/syscall.h>

namespace Poseidon {

namespace {
	volatile boost::uint64_t g_mask = (boost::uint64_t)-1;
	__thread char t_tag[5] = "----";
}

boost::uint64_t Logger::get_mask() NOEXCEPT {
	return atomic_load(g_mask, ATOMIC_RELAXED);
}
boost::uint64_t Logger::set_mask(boost::uint64_t to_disable, boost::uint64_t to_enable) NOEXCEPT {
	boost::uint64_t old_mask, new_mask;
	old_mask = atomic_load(g_mask, ATOMIC_RELAXED);
	do {
		new_mask = old_mask;
		remove_flags(new_mask, to_disable);
		add_flags(new_mask, to_enable);
	} while(!atomic_compare_exchange(g_mask, old_mask, new_mask, ATOMIC_RELAXED, ATOMIC_RELAXED));
	return old_mask;
}

bool Logger::initialize_mask_from_config(){
	const AUTO(log_masked_levels, MainConfig::get<std::string>("log_masked_levels"));
	if(log_masked_levels.empty()){
		return false;
	}
	boost::uint64_t new_mask = (boost::uint64_t)-1;
	unsigned index = 0;
	for(AUTO(it, log_masked_levels.rbegin()); (it != log_masked_levels.rend()) && (index < 64); ++it){
		switch(*it){
		case '0':
			new_mask |=  (1ull << index);
			++index;
			break;
		case '1':
			new_mask &= ~(1ull << index);
			++index;
			break;
		case ' ':
		case ',':
		case '_':
		case '-':
		case ';':
		case '/':
			break;
		default:
			throw std::invalid_argument("Invalid log_masked_levels string in main.conf");
		}
	}
	set_mask((boost::uint64_t)-1, new_mask);
	return true;
}
void Logger::finalize_mask() NOEXCEPT {
	set_mask(0, SP_POSEIDON | SP_MAJOR | LV_INFO | LV_WARNING | LV_ERROR | LV_FATAL);
}

const char *Logger::get_thread_tag() NOEXCEPT {
	return t_tag;
}
void Logger::set_thread_tag(const char *tag) NOEXCEPT {
	::snprintf(t_tag, sizeof(t_tag), "%-*s", (int)(sizeof(t_tag) - 1), tag);
}

Logger::Logger(boost::uint64_t mask, const char *file, std::size_t line) NOEXCEPT
	: m_mask(mask), m_file(file), m_line(line)
{
	m_stream <<std::boolalpha;
}
Logger::~Logger() NOEXCEPT
try {
	struct LevelElement {
		char name[16];
		unsigned char color;
		bool highlighted;
		bool to_stderr;
	};
	static CONSTEXPR const boost::array<LevelElement, 6> s_levels = {{
		{ "FATAL", '5', 1, 1 },  // brightmagenta
		{ "ERROR", '1', 1, 1 },  // brightred
		{ "WARN ", '3', 1, 1 },  // brightyellow
		{ "INFO ", '2', 0, 0 },  // green
		{ "DEBUG", '6', 0, 0 },  // cyan
		{ "TRACE", '4', 1, 0 },  // brightblue
	}};

	const unsigned level = static_cast<unsigned>(__builtin_ctzl(m_mask | LV_TRACE));
	const int output_fd = s_levels.at(level).to_stderr ? STDERR_FILENO : STDOUT_FILENO;
	const bool output_color = ::isatty(output_fd);

	StreamBuffer buf;
	char str[256];
	std::size_t len;
	// Append the timestamp in brightred (when outputting to stderr) or green (when outputting to stdout).
	if(output_color){
		buf.put("\x1B[0;3");
		if(s_levels.at(level).to_stderr){
			buf.put("1;1");
		} else {
			buf.put("2");
		}
		buf.put('m');
	}
	len = format_time(str, sizeof(str), get_local_time(), true);
	buf.put(str, len);
	if(output_color){
		buf.put("\x1B[0m");
	}
	buf.put(' ');
	// Append the thread tag in reversed red (when outputting to stderr) or yellow (when outputting to stdout).
	if(output_color){
		buf.put("\x1B[0;7;3");
		if(s_levels.at(level).to_stderr){
			buf.put("1");
		} else {
			buf.put("3");
		}
		buf.put('m');
	}
	buf.put(t_tag, sizeof(t_tag) - 1);
	if(output_color){
		buf.put("\x1B[0m");
	}
	buf.put(' ');
	// Append the thread id in red (when outputting to stderr) or yellow (when outputting to stdout).
	if(output_color){
		buf.put("\x1B[0;3");
		if(s_levels.at(level).to_stderr){
			buf.put("1");
		} else {
			buf.put("3");
		}
		buf.put('m');
	}
	len = (unsigned)std::sprintf(str, "%5lu", (unsigned long)::syscall(SYS_gettid));
	buf.put(str, len);
	if(output_color){
		buf.put("\x1B[0m");
	}
	buf.put(' ');
	// Append the level name in reversed color.
	if(output_color){
		buf.put("\x1B[0;7;3");
		buf.put(s_levels.at(level).color);
		buf.put('m');
	}
	buf.put(s_levels.at(level).name);
	if(output_color){
		buf.put("\x1B[0m");
	}
	buf.put(' ');
	// Append the log data.
	if(output_color){
		buf.put("\x1B[0;3");
		buf.put(s_levels.at(level).color);
		if(s_levels.at(level).highlighted){
			buf.put(";1");
		}
		buf.put('m');
	}
	buf.splice(m_stream.get_buffer());
	if(output_color){
		buf.put("\x1B[0m");
	}
	buf.put(' ');
	// Append the file name and line number in brightblue.
	if(output_color){
		buf.put("\x1B[0;34m");
	}
	buf.put('#');
	buf.put(m_file);
	len = (unsigned)std::sprintf(str, ":%lu", (unsigned long)m_line);
	buf.put(str, len);
	// Restore the color and end this line of log.
	if(output_color){
		buf.put("\x1B[0m");
	}
	buf.put('\n');

	static ::pthread_mutex_t s_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
	int err_code = ::pthread_mutex_lock(&s_mutex);
	(void)err_code;
	assert(err_code == 0);
	for(;;){
		len = buf.peek(str, sizeof(str));
		if(len == 0){
			break;
		}
		::ssize_t written = ::write(output_fd, str, len);
		if(written <= 0){
			break;
		}
		buf.discard(static_cast<std::size_t>(written));
	}
	err_code = ::pthread_mutex_unlock(&s_mutex);
	assert(err_code == 0);
} catch(...){
	return;
}

void Logger::put(bool val){
	m_stream <<val;
}
void Logger::put(char val){
	m_stream <<val;
}
void Logger::put(signed char val){
	m_stream <<static_cast<int>(val);
}
void Logger::put(unsigned char val){
	m_stream <<static_cast<unsigned>(val);
}
void Logger::put(short val){
	m_stream <<static_cast<int>(val);
}
void Logger::put(unsigned short val){
	m_stream <<static_cast<unsigned>(val);
}
void Logger::put(int val){
	m_stream <<val;
}
void Logger::put(unsigned val){
	m_stream <<val;
}
void Logger::put(long val){
	m_stream <<val;
}
void Logger::put(unsigned long val){
	m_stream <<val;
}
void Logger::put(long long val){
	m_stream <<val;
}
void Logger::put(unsigned long long val){
	m_stream <<val;
}
void Logger::put(const char *val){
	m_stream <<val;
}
void Logger::put(const signed char *val){
	m_stream <<static_cast<const void *>(val);
}
void Logger::put(const unsigned char *val){
	m_stream <<static_cast<const void *>(val);
}
void Logger::put(const void *val){
	m_stream <<val;
}

}
