#include "precompiled.hpp"
#include "log.hpp"
#include <boost/thread/mutex.hpp>
#include <unistd.h>
#include <time.h>
#include "atomic.hpp"
#include "utilities.hpp"
using namespace Poseidon;

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

volatile bool g_mutexInited = false; // 得对付下静态对象的构造顺序问题。
boost::mutex g_mutex;

__thread char t_tag[5] = "----";

struct MutexGuard : boost::noncopyable {
	MutexGuard(){
		atomicStore(g_mutexInited, true);
	}
	~MutexGuard(){
		atomicStore(g_mutexInited, false);
	}
} g_mutexGuard;

}

unsigned long long Logger::getMask() NOEXCEPT {
	return atomicLoad(g_mask);
}
unsigned long long Logger::setMask(unsigned long long newMask) NOEXCEPT {
	return atomicExchange(g_mask, newMask);
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
	static const bool useAsciiColors = ::isatty(STDOUT_FILENO);

	try {
		AUTO_REF(levelItem, LEVEL_ITEMS[__builtin_ctz(m_mask | LV_TRACE)]);

		char temp[256];
		const AUTO(now, getLocalTime());
		::tm desc;
		const ::time_t seconds = now / 1000;
		const unsigned milliseconds = now % 1000;
		::gmtime_r(&seconds, &desc);
		unsigned len = std::sprintf(temp, "%04u-%02u-%02u %02u:%02u:%02u.%03u ",
			1900 + desc.tm_year, 1 + desc.tm_mon, desc.tm_mday,
			desc.tm_hour, desc.tm_min, desc.tm_sec, milliseconds);

		std::string line;
		line.reserve(255);

		if(useAsciiColors){
			line += "\x1B[0;32m";
		}
		line.append(temp, len);

		if(useAsciiColors){
			line += "\x1B[0m";
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
				ch = '.';
			}
			line.push_back(ch);
		}
		line += ' ';

		if(useAsciiColors){
			line += "\x1B[0;34m";
		}
		line += '#';
		line += m_file;
		len = std::sprintf(temp, ":%lu", (unsigned long)m_line);
		line.append(temp, len);

		if(useAsciiColors){
			line += "\x1B[0m";
		}
		line += '\n';

		boost::mutex::scoped_lock lock;
		if(atomicLoad(g_mutexInited)){ // 如果为 false，则静态的 mutex 还没有被构造或者已被析构。
			boost::mutex::scoped_lock(g_mutex).swap(lock);
		}
		std::fwrite(line.data(), line.size(), sizeof(char), stdout);
	} catch(...){
	}
}
