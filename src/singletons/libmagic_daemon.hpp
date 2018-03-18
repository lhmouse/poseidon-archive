// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SYSTEM_LIBMAGIC_DAEMON_HPP_
#define POSEIDON_SYSTEM_LIBMAGIC_DAEMON_HPP_

#include <cstddef>

namespace Poseidon {

class LibMagicDaemon {
private:
	LibMagicDaemon();

public:
	static void start();
	static void stop();

	static const char *guess_mime_type(const void *data, std::size_t size);
};

}

#endif
