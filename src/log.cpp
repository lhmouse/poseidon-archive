// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "log.hpp"
#include "atomic.hpp"
#include "time.hpp"
#include "singletons/main_config.hpp"
#include <unistd.h>
#include <sys/syscall.h>

namespace Poseidon {

namespace {
	enum {
		CFG_BLACK    = 0000,
		CFG_RED      = 0001,
		CFG_GREEN    = 0002,
		CFG_YELLOW   = 0003,
		CFG_BLUE     = 0004,
		CFG_MAGNETA  = 0005,
		CFG_CYAN     = 0006,
		CFG_WHITE    = 0007,

		CFL_BRIGHT   = 0010,
		CFL_BLINKING = 0020,
		CFL_REVERSE  = 0040,
	};

	struct LevelElement {
		char name[16];
		int color;
		bool to_stderr;
	};
	CONSTEXPR const boost::array<LevelElement, 6> s_levels = {{
		{ "FATAL", CFG_MAGNETA | CFL_BRIGHT, 1 },
		{ "ERROR", CFG_RED     | CFL_BRIGHT, 1 },
		{ "WARN ", CFG_YELLOW  | CFL_BRIGHT, 1 },
		{ "INFO ", CFG_GREEN               , 0 },
		{ "DEBUG", CFG_CYAN                , 0 },
		{ "TRACE", CFG_BLUE                , 0 },
	}};

	std::size_t begin_color(char *buf, int flags){
		std::size_t len = 0;
		buf[len++] = '\x1B';
		buf[len++] = '[';
		buf[len++] = '0';
		buf[len++] = ';';
		buf[len++] = '3';
		buf[len++] = (char)('0' + (((flags & 0007) >> 0) & 7));
		if(flags & CFL_BRIGHT){
			buf[len++] = ';';
			buf[len++] = '1';
		}
		if(flags & CFL_BLINKING){
			buf[len++] = ';';
			buf[len++] = '5';
		}
		if(flags & CFL_REVERSE){
			buf[len++] = ';';
			buf[len++] = '7';
		}
		buf[len++] = 'm';
		buf[len] = 0;
		return len;
	}
	std::size_t end_color(char *buf){
		std::size_t len = 0;
		buf[len++] = '\x1B';
		buf[len++] = '[';
		buf[len++] = '0';
		buf[len++] = 'm';
		buf[len] = 0;
		return len;
	}

	volatile boost::uint64_t g_mask = (boost::uint64_t)-1;
	__thread char t_tag[5] = "----";
}

boost::uint64_t Logger::get_mask() NOEXCEPT {
	return atomic_load(g_mask, memorder_relaxed);
}
boost::uint64_t Logger::set_mask(boost::uint64_t to_disable, boost::uint64_t to_enable) NOEXCEPT {
	boost::uint64_t old_mask, new_mask;
	old_mask = atomic_load(g_mask, memorder_relaxed);
	do {
		new_mask = old_mask;
		new_mask &= ~to_disable;
		new_mask |= to_enable;
	} while(!atomic_compare_exchange(g_mask, old_mask, new_mask, memorder_relaxed, memorder_relaxed));
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
			new_mask |= (1ull << index);
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
	const unsigned level = static_cast<unsigned>(__builtin_ctzl(m_mask | LV_TRACE));
	const LevelElement *const lc = &s_levels.at(level);
	const int output_fd = lc->to_stderr ? STDERR_FILENO : STDOUT_FILENO;
	const bool output_color = ::isatty(output_fd);

	StreamBuffer buf;
	int flags;
	char str[256];
	std::size_t len;
	// Append the timestamp in brightred (when outputting to stderr) or green (when outputting to stdout).
	if(output_color){
		flags = lc->to_stderr ? (CFG_RED | CFL_BRIGHT) : CFG_GREEN;
		len = begin_color(str, flags);
		buf.put(str, len);
	}
	len = format_time(str, sizeof(str), get_local_time(), true);
	buf.put(str, len);
	if(output_color){
		len = end_color(str);
		buf.put(str, len);
	}
	buf.put(' ');
	// Append the thread tag in reverse brightred (when outputting to stderr) or yellow (when outputting to stdout).
	if(output_color){
		flags = lc->to_stderr ? (CFG_RED | CFL_BRIGHT) : CFG_YELLOW;
		flags ^= CFL_REVERSE;
		len = begin_color(str, flags);
		buf.put(str, len);
	}
	buf.put(t_tag, sizeof(t_tag) - 1);
	if(output_color){
		len = end_color(str);
		buf.put(str, len);
	}
	buf.put(' ');
	// Append the thread id in brightred (when outputting to stderr) or yellow (when outputting to stdout).
	if(output_color){
		flags = lc->to_stderr ? (CFG_RED | CFL_BRIGHT) : CFG_YELLOW;
		len = begin_color(str, flags);
		buf.put(str, len);
	}
	len = (unsigned)std::sprintf(str, "%5lu", (unsigned long)::syscall(SYS_gettid));
	buf.put(str, len);
	if(output_color){
		len = end_color(str);
		buf.put(str, len);
	}
	buf.put(' ');
	// Append the level name in reverse color.
	if(output_color){
		flags = lc->color;
		flags ^= CFL_REVERSE;
		len = begin_color(str, flags);
		buf.put(str, len);
	}
	buf.put(lc->name);
	if(output_color){
		len = end_color(str);
		buf.put(str, len);
	}
	buf.put(' ');
	// Append the log data.
	if(output_color){
		flags = lc->color;
		len = begin_color(str, flags);
		buf.put(str, len);
	}
	buf.splice(m_stream.get_buffer());
	if(output_color){
		len = end_color(str);
		buf.put(str, len);
	}
	buf.put(' ');
	// Append the file name and line number in blue.
	if(output_color){
		flags = CFG_BLUE;
		len = begin_color(str, flags);
		buf.put(str, len);
	}
	buf.put("### ");
	buf.put(m_file);
	len = (unsigned)std::sprintf(str, ":%lu", (unsigned long)m_line);
	buf.put(str, len);
	// Restore the color and end this line of log.
	if(output_color){
		len = end_color(str);
		buf.put(str, len);
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
