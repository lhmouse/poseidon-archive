#include "../precompiled.hpp"
#include "log.hpp"
#include "atomic.hpp"
#include "utilities.hpp"
#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <sys/syscall.h>
using namespace Poseidon;

namespace {

volatile unsigned g_logLevel = Log::DEBUG;
boost::mutex g_coutMutex;

}

unsigned int Log::getLevel(){
	return atomicLoad(g_logLevel);
}
void Log::setLevel(unsigned int newLevel){
	atomicStore(g_logLevel, newLevel);
}

Log::Log(unsigned level, const char *comment, const char *file, std::size_t line) throw()
	: m_level(level), m_comment(comment), m_file(file), m_line(line)
{
}
Log::~Log() throw() {
	static const char COLORS[] = { '5', '1', '3', '2', '6' };

	try {
		AUTO(const now, boost::posix_time::second_clock::local_time());
		AUTO(const str, m_stream.str());
		AUTO(const color, (m_level >= COUNT_OF(COLORS)) ? '9' : COLORS[m_level]);

		const boost::mutex::scoped_lock lock(g_coutMutex);
		std::cout <<now <<" [" <<::syscall(SYS_gettid) <<"] "
			<<"\x9B\x33" <<color << 'm'
			<<m_comment <<' ' <<m_file <<':' <<m_line <<' ' <<str
			<<"\x9B\x30m" <<std::endl;
	} catch(...){
	}
}
