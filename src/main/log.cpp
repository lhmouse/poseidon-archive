#include "../precompiled.hpp"
#include "log.hpp"
#include "atomic.hpp"
#include "utilities.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <sys/syscall.h>
using namespace Poseidon;

namespace {

volatile unsigned g_logLevel = Log::DEBUG;
boost::mutex g_cerrMutex;

}

unsigned int Log::getLevel(){
	return atomicLoad(g_logLevel);
}
void Log::setLevel(unsigned int newLevel){
	atomicStore(g_logLevel, newLevel);
}

Log::Log(unsigned level, const char *comment, const char *file, std::size_t line) NOEXCEPT
	: m_level(level), m_comment(comment), m_file(file), m_line(line)
{
}
Log::~Log() NOEXCEPT {
	static const char COLORS[] = { '5', '1', '3', '2', '6' };

	try {
		const AUTO(now, boost::posix_time::second_clock::local_time());
		const AUTO(str, m_stream.str());
		const AUTO(color, (m_level >= COUNT_OF(COLORS)) ? '9' : COLORS[m_level]);

		const boost::mutex::scoped_lock lock(g_cerrMutex);
		std::cerr <<now <<" [" <<std::setw(5) <<::syscall(SYS_gettid) <<"] "
			<<"\x9B\x33" <<color << 'm'
			<<m_comment <<' ' <<m_file <<':' <<m_line <<' ' <<str
			<<"\x9B\x30m" <<std::endl;
	} catch(...){
	}
}
