#include "../precompiled.hpp"
#include "log.hpp"
#include "utilities.hpp"
#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using namespace Poseidon;

namespace {

volatile unsigned g_logLevel = Log::INFO;
boost::mutex g_cerrMutex;

}

unsigned int Log::getLogLevel(){
	return g_logLevel;
}
void Log::setLogLevel(unsigned int newLevel){
	g_logLevel = newLevel;
}

Log::Log(const char *comment, const char *file, std::size_t line) throw()
	: m_comment(comment), m_file(file), m_line(line)
{
}
Log::~Log() throw() {
	try {
		AUTO(const now, boost::posix_time::second_clock::local_time());
		m_stream <<std::endl;
		AUTO(const str, m_stream.str());

		const boost::mutex::scoped_lock lock(g_cerrMutex);
		std::cerr <<now <<' ' <<std::setw(8) <<std::left <<m_comment
			<<m_file <<':' <<m_line <<' ' <<str;
	} catch(...){
	}
}
