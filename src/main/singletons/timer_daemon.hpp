#ifndef POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_
#define POSEIDON_SINGLETONS_TIMER_DAEMON_HPP_

namespace Poseidon {

struct TimerDaemon {
	static void start();
	static void stop();

private:
	TimerDaemon();
};

}

#endif
