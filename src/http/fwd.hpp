// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_FWD_HPP_
#define POSEIDON_HTTP_FWD_HPP_

namespace Poseidon {

namespace Http {
	class RequestHeaders;
	class ResponseHeaders;
	class Exception;

	class ServerReader;
	class ServerWriter;
	class ClientReader;
	class ClientWriter;

	class Session;
	class Client;
	class UpgradedSessionBase;
}

}

#endif
