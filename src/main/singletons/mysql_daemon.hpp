#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include <boost/shared_ptr.hpp>

namespace Poseidon {

struct MySqlDaemon {
	static void start();
	static void stop();

private:
	MySqlDaemon();
};

}

#endif
