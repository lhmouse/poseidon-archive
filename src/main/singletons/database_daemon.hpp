#ifndef POSEIDON_DATABASE_DAEMON_HPP_
#define POSEIDON_DATABASE_DAEMON_HPP_

namespace Poseidon {

struct DatabaseDaemon {
	static void start();
	static void stop();

private:
	DatabaseDaemon();
};

}

#endif
