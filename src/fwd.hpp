// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FWD_HPP_
#define POSEIDON_FWD_HPP_

namespace Poseidon {

class TinyException;
class Exception;
class ProtocolException;
class SystemException;

class Thread;
class Mutex;
class RecursiveMutex;
class ConditionVariable;

class OptionalMap;
class StreamBuffer;
class HexPrinter;
class SharedNts;
class IpPort;
class SockAddr;
class Uuid;
class VirtualSharedFromThis;

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

class ConfigFile;
class CsvParser;
class JsonObject;
class JsonArray;
class JsonElement;

class JobBase;
class EventBase;
class SystemServletBase;

class Promise;
template<typename> class PromiseContainer;

class SocketBase;
class SessionBase;
class TcpSessionBase;
class TcpClientBase;
class TcpServerBase;
class UdpServerBase;

class SystemSession;

class HexEncoder;
class HexDecoder;
class Base64Encoder;
class Base64Decoder;

class Deflator;
class Inflator;

class EventListener;
class Timer;

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
class SystemServer;
class TimerDaemon;
class WorkhorseCamp;

}

#endif
