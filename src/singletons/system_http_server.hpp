// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

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
