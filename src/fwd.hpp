// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FWD_HPP_
#define POSEIDON_FWD_HPP_

namespace Poseidon {

class Thread;
class Logger;
class Profiler;
class Exception;
class ProtocolException;
class SystemException;

class OptionalMap;
class StreamBuffer;
class SharedNts;
class IpPort;
class SockAddr;
class SessionBase;
class Transaction;
class DateTime;
class HexDumper;
class Uuid;
class VirtualSharedFromThis;

class ConfigFile;
class CsvParser;
class JsonObject;
class JsonArray;

class JobBase;
class EventBaseWithoutId;

class Epoll;
class SocketServerBase;
class TcpSessionBase;
class TcpClientBase;
class TcpServerBase;
class UdpServerBase;

class TimerItem;
class Module;
class EventListener;

}

#endif
