// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FWD_HPP_
#define POSEIDON_FWD_HPP_

namespace Poseidon {

class Exception;
class ProtocolException;
class SystemException;

class Thread;
class Mutex;
class RecursiveMutex;
class ConditionVariable;

class OptionalMap;
class StreamBuffer;
class SharedNts;
class IpPort;
class SockAddr;
class Uuid;
class VirtualSharedFromThis;

class Buffer_streambuf;
class Buffer_istream;
class Buffer_ostream;
class Buffer_iostream;

class ConfigFile;
class CsvParser;
class JsonObject;
class JsonArray;
class JsonElement;

class JobBase;
class JobPromise;
class EventBaseWithoutId;

class SessionBase;
class Epoll;
class SocketServerBase;
class TcpSessionBase;
class TcpClientBase;
class TcpServerBase;
class UdpServerBase;

class Deflator;
class Inflator;

class TimerItem;
class Module;
class EventListener;

class DnsDaemon;
class EpollDaemon;
class EventDispatcher;
class FileSystemDaemon;
class JobDispatcher;
class MainConfig;
class ModuleDepository;
class MongoDbDaemon;
class MySqlDaemon;
class ProfileDepository;
class SystemHttpServer;
class TimerDaemon;

}

#endif
