// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FWD_HPP_
#define POSEIDON_FWD_HPP_

namespace Poseidon {

class Tiny_exception;
class Exception;
class System_exception;

class Option_map;
class Stream_buffer;
class Hex_printer;
class Rcnts;
class Ip_port;
class Sock_addr;
class Uuid;
class Virtual_shared_from_this;

class Buffer_streambuf;
class Buffer_istream;
class Buffer_ostream;
class Buffer_iostream;

class Crc32_streambuf;
class Crc32_ostream;
class Md5_streambuf;
class Md5_ostream;
class Sha1_streambuf;
class Sha1_ostream;
class Sha256_streambuf;
class Sha256_ostream;

class Config_file;
class Csv_parser;
class Json_object;
class Json_array;
class Json_element;

class Job_base;
class Event_base;
class System_http_servlet_base;

class Promise;
template<typename> class Promise_container;

class Socket_base;
class Session_base;
class Tcp_session_base;
class Tcp_client_base;
class Tcp_server_base;
class Udp_session_base;
class Udp_client_base;
class Udp_server_base;

class System_http_session;

class Hex_encoder;
class Hex_decoder;
class Base64_encoder;
class Base64_decoder;

class Deflator;
class Inflator;

class Event_listener;
class Timer;

class Dns_daemon;
class Epoll_daemon;
class Event_dispatcher;
class File_system_daemon;
class Job_dispatcher;
class Main_config;
class Module_depository;
class Mongodb_daemon;
class Mysql_daemon;
class Profile_depository;
class System_http_server;
class Timer_daemon;
class Workhorse_camp;
class Magic_daemon;

}

#endif
