#include "../precompiled.hpp"
#include "log.hpp"
#include "atomic.hpp"
#include "utilities.hpp"
#include <cstdio>
#include <boost/thread/mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <unistd.h>
#include <time.h>
using namespace Poseidon;

namespace {

volatile unsigned g_logLevel = 100;
boost::mutex g_cerrMutex;

__thread unsigned t_tag;

}

unsigned int Log::getLevel(){
	return atomicLoad(g_logLevel);
}
void Log::setLevel(unsigned int newLevel){
	atomicStore(g_logLevel, newLevel);
}

unsigned Log::getThreadTag(){
	return t_tag;
}
void Log::setThreadTag(unsigned newTag){
	t_tag = newTag;
}

Log::Log(unsigned level, const char *comment, const char *file, std::size_t line) NOEXCEPT
	: m_level(level), m_comment(comment), m_file(file), m_line(line)
{
}
Log::~Log() NOEXCEPT {
	static const char COLORS[] = { '5', '1', '3', '2', '6' };
	static const char TAGS[][8] = { "P   ", " M  ", "  T ", "   E" };

	try {
		const bool withColors = ::isatty(STDERR_FILENO);
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
		for(;;){
			const std::size_t count = m_stream.readsome(temp, COUNT_OF(temp));
			if(count == 0){
				break;
			}
			line.append(temp, count);
		}
		line += '\n';
		if(withColors){
			line += "\x1B[0m";
		}

		const boost::mutex::scoped_lock lock(g_cerrMutex);
		std::fwrite(line.c_str(), line.size(), sizeof(char), stderr);
	} catch(...){
	}
}
