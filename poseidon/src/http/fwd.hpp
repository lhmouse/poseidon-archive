// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_FWD_HPP_
#define POSEIDON_HTTP_FWD_HPP_

namespace Poseidon {
namespace Http {

class Request_headers;
class Response_headers;
class Url_param;
class Header_option;
class Exception;

class Authentication_context;
class Multipart;

class Server_reader;
class Server_writer;
class Client_reader;
class Client_writer;

class Session;
class Client;
class Upgraded_session_base;

}
}

#endif
