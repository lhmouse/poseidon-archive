// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "log.hpp"
#include <unistd.h>
#include <pthread.h>
#include "atomic.hpp"
#include "time.hpp"
#include "flags.hpp"

namespace Poseidon {

namespace {
	struct LevelItem {
		char text[16];
		char color;
		bool highlighted;
	};

	const LevelItem LEVEL_ITEMS[] = {
		{ "FATAL", '5', 1 },	// 粉色
		{ "ERROR", '1', 1 },	// 红色
		{ "WARN ", '3', 1 },	// 黄色
		{ "INFO ", '2', 0 },	// 绿色
		{ "DEBUG", '6', 0 },	// 青色
		{ "TRACE", '4', 1 },	// 亮蓝
	};

	volatile unsigned long long g_mask = -1ull;

	// 不要使用 Mutex 对象。如果在其他静态对象的构造函数中输出日志，这个对象可能还没构造。
	::pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

	__thread char t_tag[5] = "----";
}

unsigned long long Logger::getMask() NOEXCEPT {
	return atomicLoad(g_mask, ATOMIC_ACQUIRE);
}
unsigned long long Logger::setMask(unsigned long long toDisable, unsigned long long toEnable) NOEXCEPT {
	unsigned long long oldMask = atomicLoad(g_mask, ATOMIC_ACQUIRE), newMask;
	do {
		newMask = oldMask;
		removeFlags(newMask, toDisable);
		addFlags(newMask, toEnable);
	} while(!atomicCompareExchange(g_mask, oldMask, newMask, ATOMIC_ACQ_REL, ATOMIC_ACQUIRE));
	return oldMask;
}

const char *Logger::getThreadTag() NOEXCEPT {
	return t_tag;
}
void Logger::setThreadTag(const char *newTag) NOEXCEPT {
	unsigned i = 0;
	while(i < sizeof(t_tag)){
		const char ch = newTag[i];
		if(ch == 0){
			break;
		}
		t_tag[i++] = ch;
	}
	while(i < sizeof(t_tag)){
		t_tag[i++] = ' ';
	}
}

Logger::Logger(unsigned long long mask, const char *file, std::size_t line) NOEXCEPT
	: m_mask(mask), m_file(file), m_line(line)
{
}
Logger::~Logger() NOEXCEPT {
	static const bool stderrUsesAsciiColors = ::isatty(STDERR_FILENO);
	static const bool stdoutUsesAsciiColors = ::isatty(STDOUT_FILENO);

	try {
		bool useAsciiColors;
		int fd;
		if(m_mask & SP_MAJOR){
			useAsciiColors = stderrUsesAsciiColors;
			fd = STDERR_FILENO;
		} else {
			useAsciiColors = stdoutUsesAsciiColors;
			fd = STDOUT_FILENO;
		}

		AUTO_REF(levelItem, LEVEL_ITEMS[__builtin_ctz(m_mask | LV_TRACE)]);

		char temp[256];
		unsigned len;

		std::string line;
		line.reserve(255);

		if(useAsciiColors){
			line += "\x1B[0;32m";
		}
		len = formatTime(temp, sizeof(temp), getLocalTime(), true);
		line.append(temp, len);

		if(useAsciiColors){
			line += "\x1B[0;33m";
		}
		len = (unsigned)std::sprintf(temp, " %02X ", (unsigned)((m_mask >> 8) & 0xFF));
		line.append(temp, len);

		if(useAsciiColors){
			line += "\x1B[0;39m";
		}
		line += '[';
		line.append(t_tag, sizeof(t_tag) - 1);
		line += ']';
		line += ' ';

		if(useAsciiColors){
			line +="\x1B[0;30;4";
			line += levelItem.color;
			line += 'm';
		}
		line += levelItem.text;
		if(useAsciiColors){
			line +="\x1B[0;40;3";
			line += levelItem.color;
			if(levelItem.highlighted){
				line += ';';
				line += '1';
			}
			line += 'm';
		}
		line += ' ';

		char ch;
		while(m_stream.get(ch)){
			if(((unsigned char)ch + 1 <= 0x20) || (ch == 0x7F)){
				ch = ' ';
			}
			line.push_back(ch);
		}
		line += ' ';

		if(useAsciiColors){
			line += "\x1B[0;34m";
		}
		line += '#';
		line += m_file;
		len = (unsigned)std::sprintf(temp, ":%lu", (unsigned long)m_line);
		line.append(temp, len);

		if(useAsciiColors){
			line += "\x1B[0m";
		}
		line += '\n';

		{
			::pthread_mutex_lock(&g_mutex);

			std::size_t bytesTotal = 0;
			while(bytesTotal < line.size()){
				const AUTO(bytesWritten, ::write(fd, line.data() + bytesTotal, line.size() - bytesTotal)); // noexcept
				if(bytesWritten <= 0){
					break;
				}
				bytesTotal += static_cast<std::size_t>(bytesWritten);
			}

			::pthread_mutex_unlock(&g_mutex);
		}
	} catch(...){
	}
}

}
