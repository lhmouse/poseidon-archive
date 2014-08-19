#include "../precompiled.hpp"
#include "log.hpp"
#include "utilities.hpp"
#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using namespace Poseidon;

namespace {

volatile unsigned g_logLevel = Log::DEBUG;
boost::mutex g_coutMutex;

}

unsigned int Log::getLogLevel(){
	return g_logLevel;
}
void Log::setLogLevel(unsigned int newLevel){
	g_logLevel = newLevel;
}

Log::Log(unsigned level, const char *comment, const char *file, std::size_t line) throw()
	: m_level(level), m_comment(comment), m_file(file), m_line(line)
{
}
Log::~Log() throw() {
	static const char COLORS[][4] = { "35", "31", "33", "32", "36" };

	try {
		AUTO(const now, boost::posix_time::second_clock::local_time());
		AUTO(const str, m_stream.str());

		const boost::mutex::scoped_lock lock(g_coutMutex);
		std::cout <<"\x9B" <<COLORS[m_level] << 'm'
			<<now <<' ' <<std::setw(8) <<std::left <<m_comment
			<<m_file <<':' <<m_line <<' ' <<str
			<<"\x9B\x30m" <<std::endl;
	} catch(...){
	}
}
