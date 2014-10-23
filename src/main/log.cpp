#include "../precompiled.hpp"
#include "log.hpp"
#include "atomic.hpp"
#include "utilities.hpp"
#include <boost/thread/once.hpp>
#include <boost/thread/mutex.hpp>
#include <unistd.h>
#include <time.h>
using namespace Poseidon;

namespace {

volatile unsigned g_logLevel = 100;
boost::once_flag g_mutexInitFlag;
boost::scoped_ptr<boost::mutex> g_mutex;

__thread unsigned t_tag;

void initMutex(){
	g_mutex.reset(new boost::mutex);
}

}

unsigned int Logger::getLevel(){
	return atomicLoad(g_logLevel);
}
void Logger::setLevel(unsigned int newLevel){
	atomicStore(g_logLevel, newLevel);
}

unsigned Logger::getThreadTag(){
	return t_tag;
}
void Logger::setThreadTag(unsigned newTag){
	t_tag = newTag;
}

Logger::Logger(unsigned level, const char *comment, const char *file, std::size_t line) NOEXCEPT
	: m_level(level), m_comment(comment), m_file(file), m_line(line)
{
}
Logger::~Logger() NOEXCEPT {
	static const char COLORS[] = { '5', '1', '3', '2', '6' };
	static const char TAGS[][8] = { "P   ", " M  ", "  T ", "   E" };

	static const bool withColors = ::isatty(STDOUT_FILENO);

	try {
		const AUTO(color, (m_level >= COUNT_OF(COLORS)) ? '9' : COLORS[m_level]);
		const AUTO(tag, (t_tag >= COUNT_OF(TAGS)) ? "" : TAGS[t_tag]);

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
		line.assign(temp, len);

		line += '[';
		line += tag;
		line += ']';
		line += ' ';

		if(withColors){
			line +="\x1B[30;4";
			line += color;
			line += 'm';
		}
		len = std::strlen(m_comment);
		assert(len <= MAX_COMMENT_WIDTH);
		const unsigned right = (MAX_COMMENT_WIDTH - len) / 2;
		const unsigned left = MAX_COMMENT_WIDTH - len - right;
		line.append(left, ' ');
		line.append(m_comment, len);
		line.append(right, ' ');
		if(withColors){
			line +="\x1B[40;3";
			line += color;
			line += 'm';
		}
		line += ' ';
		line += m_file;
		len = std::sprintf(temp, ":%lu ", (unsigned long)m_line);
		line.append(temp, len);
		char ch;
		while(m_stream.get(ch)){
			if(((unsigned char)ch + 1 <= 0x20) || (ch == 0x7F)){
				ch = '.';
			}
			line.push_back(ch);
		}
		if(withColors){
			line += "\x1B[0m";
		}
		line += '\n';

   		{
	    	boost::call_once(&initMutex, g_mutexInitFlag);
			boost::mutex::scoped_lock lock;
	    	if(g_mutex){
				boost::mutex::scoped_lock(*g_mutex).swap(lock);
			}
			std::fwrite(line.data(), line.size(), sizeof(char), stdout);
		}
	} catch(...){
	}
}
