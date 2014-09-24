#include "../precompiled.hpp"
#include "log.hpp"
#include "atomic.hpp"
#include "utilities.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <unistd.h>
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
	static const char TAGS[][8] = { "P   ", " E  ", "  T ", "   D" };

	try {
		const bool withColor = ::isatty(STDERR_FILENO);
		const AUTO(color, (m_level >= COUNT_OF(COLORS)) ? '9' : COLORS[m_level]);
		const AUTO(tag, (t_tag >= COUNT_OF(TAGS)) ? "" : TAGS[t_tag]);

		std::string line;
		line.reserve(255);

		line += '[';
		line += tag;
		line += ']';
		line += ' ';

		if(withColor){
			line +="\x1B[3";
			line += color;
			line += 'm';
		}
		line += m_comment;
		line += ' ';
		line += m_file;
		line += ':';
		line += boost::lexical_cast<std::string>(m_line);
		line += ' ';
		line += m_stream.str();
		if(withColor){
			line += "\x1B[0m";
		}

		const boost::mutex::scoped_lock lock(g_cerrMutex);
		std::cerr <<boost::posix_time::second_clock::local_time() <<' ' <<line <<std::endl;
	} catch(...){
	}
}
