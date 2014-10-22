#ifndef POSEIDON_SYSTEM_HTTP_SERVER_HPP_
#define POSEIDON_SYSTEM_HTTP_SERVER_HPP_

namespace Poseidon {

struct SystemHttpServer {
	static void start();
	static void stop();

private:
	SystemHttpServer();
};

}

#endif
