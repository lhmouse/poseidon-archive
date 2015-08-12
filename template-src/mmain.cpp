// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

// 这个文件被置于公有领域（public domain）。
/*
#include "../src/precompiled.hpp"
#include "../src/log.hpp"
#include "../src/exception.hpp"
#include "../src/profiler.hpp"
#include "../src/hash.hpp"
#include "../src/module_raii.hpp"
#include "../src/singletons/module_depository.hpp"
#include "../src/singletons/event_dispatcher.hpp"
#include "../src/singletons/timer_daemon.hpp"
#include "../src/singletons/profile_depository.hpp"
#include "../src/websocket/session.hpp"
#include "../src/http/utilities.hpp"
#include "../src/tcp_client_base.hpp"
#include "../src/cbpp/session.hpp"
#include "../src/http/session.hpp"
#include "../src/cbpp/message_base.hpp"
#include "../src/singletons/epoll_daemon.hpp"
#include "../src/mysql/object_base.hpp"
#include "../src/job_base.hpp"
#include "../src/uuid.hpp"
#include "../src/async_job.hpp"
using namespace Poseidon;

#define MYSQL_OBJECT_NAME	MySqlObj
#define MYSQL_OBJECT_FIELDS	\
	FIELD_SMALLINT(si)	\
	FIELD_STRING(str)	\
	FIELD_BIGINT(bi)	\
	FIELD_DATETIME(dt)
#include "../src/mysql/object_generator.hpp"

namespace {
	void loadedProc(bool found){
		LOG_POSEIDON_FATAL("-- loaded! found = ", found);
	}

	void write(){
		AUTO(obj, boost::make_shared<MySqlObj>());
		obj->enableAutoSaving();
		obj->set_si(999);
		obj->set_str("meow");
		for(int i = 0; i < 10; ++i){
			obj->set_bi(i);
		}
		obj->asyncLoad("SELECT * FROM `MySqlObj`", loadedProc);
	}
}
MODULE_RAII {
	enqueueAsyncJob(write, 3000);
	return VAL_INIT;
}

namespace {

struct TestEvent1 : public EventBase<1> {
	int i;
	std::string s;
};

struct TestEvent2 : public EventBase<1> {
	double d;
};

void event1Proc(boost::shared_ptr<TestEvent1> event){
	PROFILE_ME;
	LOG_POSEIDON_FATAL("event1Proc: i = ", event->i, ", s = ", event->s);
}

void event2Proc(boost::shared_ptr<TestEvent2> event){
	PROFILE_ME;
	LOG_POSEIDON_FATAL("event2Proc: d = ", event->d);
}

void printObjs(std::vector<boost::shared_ptr<MySqlObjFectBase> > v){
	LOG_POSEIDON_FATAL("--------- v.size() = ", v.size());
	for(AUTO(it, v.begin()); it != v.end(); ++it){
		AUTO(p, static_cast<MySqlObjF *>(it->get()));
		LOG_POSEIDON_FATAL("-- si = ", p->get_si(), ", str = ", p->get_str(), ", bi = ", p->get_bi());
	}
}

void mysqlExceptProc(){
	LOG_POSEIDON_FATAL("MySQL exception!");
}

void tickProc(unsigned long long now, unsigned long long period){
	PROFILE_ME;
	LOG_POSEIDON_FATAL("Tick, now = ", now, ", period = ", period);

	MySqlObjF::batchLoad("SELECT * FROM `MySqlObjF`", &printObjs, &mysqlExceptProc);
}

void profileProc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	OptionalMap headers;
	StreamBuffer contents;

	headers.set("Content-Type", "text/plain");
	contents.put("   Samples      Total time(us)  Exclusive time(us)    File:Line\n");
	const AUTO(profile, ProfileDepository::snapshot());
	for(AUTO(it, profile.begin()); it != profile.end(); ++it){
		char temp[128];
		unsigned len = (unsigned)std::sprintf(temp, "%10llu%20llu%20llu    ",
			it->samples, it->usTotal, it->usExclusive);
		contents.put(temp, len);
		contents.put(it->file);
		len = (unsigned)std::sprintf(temp, ":%lu\n", it->line);
		contents.put(temp, len);
	}
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

void meowProc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	AUTO(obj, boost::make_shared<MySqlObjF>());
	obj->set_si(123);
	obj->set_str("meow");
	obj->set_bi(456789);
	obj->asyncSave(true);

	OptionalMap headers;
	StreamBuffer contents;

	AUTO(event, boost::make_shared<TestEvent1>());
	event->i = 123;
	event->s = "meow";
	event->raise();

	headers.set("Content-Type", "text/html");
	contents.put("<h1>Meow!</h1>");
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}
void meowMeowProc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	OptionalMap headers;
	StreamBuffer contents;

	AUTO(event, boost::make_shared<TestEvent1>());
	event->i = 123;
	event->s = "meow/meow";
	event->raise();

	headers.set("Content-Type", "text/html");
	contents.put("<h1>Meow! Meow!</h1>");
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

boost::weak_ptr<std::vector<boost::shared_ptr<void> > > g_servlets;

void loadProc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	OptionalMap headers;
	StreamBuffer contents;

	AUTO(event, boost::make_shared<TestEvent2>());
	event->d = 123.45;
	event->raise();

	AUTO_REF(v, *boost::shared_ptr<std::vector<boost::shared_ptr<void> > >(g_servlets));
	if(!v.empty()){
		contents.put("Already loaded");
	} else {
		v.push_back(HttpServletDepository::create(1, SSLIT("/meow/"), &meowProc));
		v.push_back(HttpServletDepository::create(1, SSLIT("/meow/meow/"), &meowMeowProc));
		v.push_back(TimerDaemon::registerTimer(5000, 10000, &tickProc));
		contents.put("OK");
	}
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}
void unloadProc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	OptionalMap headers;
	StreamBuffer contents;

	AUTO(event, boost::make_shared<TestEvent2>());
	event->d = 67.89;
	event->raise();

	AUTO_REF(v, *boost::shared_ptr<std::vector<boost::shared_ptr<void> > >(g_servlets));
	if(v.empty()){
		contents.put("Already unloaded");
	} else {
		v.clear();
		contents.put("OK");
	}
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

void webSocketProc(boost::shared_ptr<WebSocketSession> wss,
	WebSocketOpCode opcode, StreamBuffer incoming)
{
	PROFILE_ME;
	LOG_POSEIDON_FATAL("Received packet: opcode = ", opcode, ", payload = ", StreamBufferHexDumper(incoming));

	std::string s;
	incoming.dump(s);

	char crc32Str[16];
	std::sprintf(crc32Str, "%08lx", (unsigned long)crc32Sum(s.data(), s.size()));

	unsigned char md5[16];
	md5Sum(md5, s.data(), s.size());
	std::string md5Str = hexEncode(md5, sizeof(md5));

	unsigned char sha1[20];
	sha1Sum(sha1, s.data(), s.size());
	std::string sha1Str = hexEncode(sha1, sizeof(sha1));

	StreamBuffer out;
	out.put("CRC32: ");
	out.put(crc32Str);
	out.put("\nMD5: ");
	out.put(md5Str.data(), md5Str.size());
	out.put("\nSHA1: ");
	out.put(sha1Str.data(), sha1Str.size());
	out.put('\n');
	wss->send(STD_MOVE(out), false);
}

class TestClient : public TcpClientBase {
public:
	static boost::shared_ptr<TestClient> create(){
		AUTO(ret, boost::make_shared<TestClient>());
		ret->goResident();
		return ret;
	}

public:
	TestClient()
		: TcpClientBase(IpPort(SSLIT("192.30.252.128"), 443), true)
	{
	}

private:
	void onReadAvail(const void *data, std::size_t size){
		AUTO(read, (const char *)data);
		for(std::size_t i = 0; i < size; ++i){
			std::putchar(read[i]);
		}
	}
};

}

#define MESSAGE_NAME		TestInt
#define MESSAGE_ID			100
#define MESSAGE_FIELDS		FIELD_VINT(i)
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME		TestUInt
#define MESSAGE_ID			101
#define MESSAGE_FIELDS 	FIELD_VUINT(u)
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME		TestString
#define MESSAGE_ID			102
#define MESSAGE_FIELDS		FIELD_STRING(s)
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME		TestIntArray
#define MESSAGE_ID			103
#define MESSAGE_FIELDS		FIELD_ARRAY(a, FIELD_VINT(i))
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME		TestUIntArray
#define MESSAGE_ID			104
#define MESSAGE_FIELDS		FIELD_ARRAY(a, FIELD_VUINT(u))
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME		TestStringArray
#define MESSAGE_ID			105
#define MESSAGE_FIELDS		FIELD_ARRAY(a, FIELD_STRING(s))
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME	   	TestMessage
#define MESSAGE_ID			106
#define MESSAGE_FIELDS \
	FIELD_VINT(i)   \
	FIELD_VUINT(j)  \
	FIELD_ARRAY(a,  \
		FIELD_BYTES(b, 3)	\
		FIELD_STRING(s) \
		FIELD_VUINT(k)  \
	)
#include "../src/cbpp/message_generator.hpp"

namespace {

void TestIntProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestInt req(incoming);
	LOG_POSEIDON_WARNING("sint = ", req.i);
	req.i /= 10;
	ps->send(1000, req);
}
void TestUIntProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestUInt req(incoming);
	LOG_POSEIDON_WARNING("int = ", req.u);
	req.u /= 10;
	ps->send(1001, req);
}
void TestStringProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestString req(incoming);
	LOG_POSEIDON_WARNING("string = ", req.s);
	req.s += "_0123456789";
	ps->send(1002, req);
}

void TestIntArrayProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestIntArray req(incoming);
	LOG_POSEIDON_WARNING("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARNING("  ", i, " = ", req.a.at(i).i);
		req.a.at(i).i /= 10;
	}
	ps->send(1003, req);
}
void TestUIntArrayProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestUIntArray req(incoming);
	LOG_POSEIDON_WARNING("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARNING("  ", i, " = ", req.a.at(i).u);
		req.a.at(i).u /= 10;
	}
	ps->send(1004, req);
}
void TestStringArrayProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestStringArray req(incoming);
	LOG_POSEIDON_WARNING("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARNING("  ", i, " = ", req.a.at(i).s);
		req.a.at(i).s += "_0123456789";
	}
	ps->send(1005, req);
}

void TestProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	LOG_POSEIDON_WARNING("Received: ", HexDumper(incoming));
	TestMessage req(incoming);
	LOG_POSEIDON_WARNING("req.i = ", req.i);
	LOG_POSEIDON_WARNING("req.j = ", req.j);
	LOG_POSEIDON_WARNING("req.a.size() = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARNING("req.a[", i, "].s = ", req.a.at(i).s);
		LOG_POSEIDON_WARNING("req.a[", i, "].k = ", req.a.at(i).k);
	}
	ps->send(1006, req);
}

}

MODULE_RAII {
	return HttpServletDepository::create(1, SSLIT("/profile"), &profileProc);
}
MODULE_RAII {
	return HttpServletDepository::create(1, SSLIT("/load"), &loadProc);
}
MODULE_RAII {
	return HttpServletDepository::create(1, SSLIT("/unload"), &unloadProc);
}
MODULE_RAII {
	AUTO(v, boost::make_shared<std::vector<boost::shared_ptr<void> > >());
	g_servlets = v;
	return v;
}
MODULE_RAII {
	return EventDispatcher::registerListener<TestEvent1>(&event1Proc);
}
MODULE_RAII {
	return EventDispatcher::registerListener<TestEvent2>(&event2Proc);
}
MODULE_RAII {
	return WebSocketServletDepository::create(2, SSLIT("/wstest"), &webSocketProc);
}

MODULE_RAII {
	return CbppServletDepository::create(2, 100, &TestIntProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 101, &TestUIntProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 102, &TestStringProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 103, &TestIntArrayProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 104, &TestUIntArrayProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 105, &TestStringArrayProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 106, &TestProc);
}

namespace {

void onClientClose(int id){
	LOG_POSEIDON_FATAL("Client closed! ", id);
}

}

MODULE_RAII {

	LOG_POSEIDON_INFO("Connecting to github...");
	AUTO(p, TestClient::create());
	p->registerOnClose(boost::bind(&onClientClose, 0));
	p->registerOnClose(boost::bind(&onClientClose, 1));
	p->registerOnClose(boost::bind(&onClientClose, 2));
	p->registerOnClose(boost::bind(&onClientClose, 3));
	p->send(StreamBuffer("GET / HTTP/1.1\r\nHost: github.com\r\n\r\n"));

	AUTO(obj, boost::make_shared<MySqlObjF>());
	obj->set_si(999);
	obj->set_str("meow");
	obj->set_bi(456789);
	obj->asyncSave(false);

	TestMessage req;
	req.i = 12345;
	req.j = 54321;
	req.a.resize(2);
	req.a.at(0).s = "meow";
	req.a.at(0).k = 56789;
	std::memcpy(req.a.at(0).b, "ABC", 3);
	req.a.at(1).s = "bark";
	req.a.at(1).k = 98765;
	std::memcpy(req.a.at(1).b, "DEF", 3);
	LOG_POSEIDON_FATAL(req);
	return VAL_INIT;
}

MODULE_RAII {
	AUTO(server, (boost::make_shared<CbppServer>(2,
		IpPort(SharedNts("0.0.0.0"), 8850), NULLPTR, NULLPTR)));
	EpollDaemon::registerServer(server);
	return server;
}
MODULE_RAII {
	AUTO(server, (boost::make_shared<HttpServer>(2,
		IpPort(SharedNts("0.0.0.0"), 8860), NULLPTR, NULLPTR, std::vector<std::string>())));
	EpollDaemon::registerServer(server);
	return server;
}

MODULE_RAII {
	std::set<Uuid> s;
	for(unsigned i = 0; i < 1000000; ++i){
		s.insert(Uuid::generate());
	}
	LOG_POSEIDON_FATAL("number of uuid generated: ", s.size());
	LOG_POSEIDON_FATAL("first: ", *s.begin());
}

MODULE_RAII {
	LOG_POSEIDON_FATAL("----------- ", explode<std::string>(':', "0:1:2:3:").size());
	return VAL_INIT;
}

namespace {
	// const AUTO(sp, boost::make_shared<int>());
	boost::shared_ptr<int> sp;

	class MyJob : public JobBase {
	private:
		const char *const m_s;

	public:
		explicit MyJob(const char *s)
			: m_s(s)
		{
		}

	public:
		boost::weak_ptr<const void> getCategory() const OVERRIDE {
			return sp;
		}
		void perform() const OVERRIDE {
			LOG_POSEIDON_FATAL("!!!! MyJob::perform: ", m_s);
			suspendCurrentJob(VAL_INIT);
		}
	};

	class MeowJob : public JobBase {
	public:
		boost::weak_ptr<const void> getCategory() const OVERRIDE {
			return VAL_INIT;
		}
		void perform() const OVERRIDE {
			LOG_POSEIDON_FATAL("!!!! MeowJob::perform");
		}
	};
}

MODULE_RAII {
	AUTO(withdrawn, boost::make_shared<bool>(true));
	enqueueJob(boost::make_shared<MyJob>("1  "), 0, withdrawn);
	enqueueJob(boost::make_shared<MyJob>(" 2 "), 0, withdrawn);
	enqueueJob(boost::make_shared<MyJob>("  3"), 0, withdrawn);

	enqueueJob(boost::make_shared<MeowJob>());

	LOG_POSEIDON_FATAL("Job enqueued!");
	return VAL_INIT;
}

namespace {

void meowProc(boost::shared_ptr<Http::Session> session, Http::Request req){
	PROFILE_ME;

	LOG_POSEIDON_FATAL("Contents = ", req.contents);

	OptionalMap headers;
	headers.set("Content-Type", "text/html");
	StreamBuffer contents;
	contents.put("<h1>Meow!</h1>");
	session->send(Http::ST_OK, STD_MOVE(headers), STD_MOVE(contents));
}

}

MODULE_RAII {
	AUTO(server, boost::make_shared<Http::Server>(2,
		IpPort(SharedNts("0.0.0.0"), 8860), NULLPTR, NULLPTR, std::vector<std::string>()));
	EpollDaemon::registerServer(server);
	return server;
}
MODULE_RAII {
	return HttpServletDepository::create(2, SSLIT("/meow"), &meowProc);
}


#include "../src/precompiled.hpp"
#include "../src/http/client.hpp"
#include "../src/http/utilities.hpp"
#include "../src/log.hpp"
#include "../src/module_raii.hpp"

namespace {

using namespace Poseidon;

class MyClient : public Http::Client {
public:
	static boost::shared_ptr<MyClient> create(){
		boost::shared_ptr<MyClient> ret(new MyClient);
		ret->goResident();

		OptionalMap headers;
		headers.set("Host", "github.com");
//		headers.set("Transfer-Encoding", "chunked");
//		headers.set("Connection", "Close");
		ret->send(Http::V_GET, "/", OptionalMap(), STD_MOVE(headers), StreamBuffer("test"));
//		ret->shutdownWrite();

		return ret;
	}

private:
	boost::uint64_t m_contentLength;
	StreamBuffer m_contents;

private:
	MyClient()
		: Http::Client(IpPort(SSLIT("192.30.252.131"), 443), true)
	{
		LOG_POSEIDON_FATAL("MyClient::MyClient()");
	}

public:
	~MyClient(){
		LOG_POSEIDON_FATAL("MyClient::~MyClient()");
	}

private:
	void print() const {
		LOG_POSEIDON_INFO("Complete content:\n", m_contents.dump());
	}

protected:
	void onResponseHeaders(const Http::ResponseHeaders &responseHeaders, boost::uint64_t contentLength) OVERRIDE {
		LOG_POSEIDON_DEBUG("onResponseHeaders(): statusCode = ", static_cast<unsigned>(responseHeaders.statusCode),
			", contentLength = ", static_cast<boost::int64_t>(contentLength));
		for(AUTO(it, responseHeaders.headers.begin()); it != responseHeaders.headers.end(); ++it){
			LOG_POSEIDON_DEBUG("> ", it->first, " = ", Http::urlDecode(it->second));
		}
		m_contentLength = contentLength;
		m_contents.clear();
	}
	void onEntity(boost::uint64_t contentOffset, const StreamBuffer &entity) OVERRIDE {
		LOG_POSEIDON_DEBUG("onEntity(): contentOffset = ", contentOffset, ", size = ", entity.size());
		AUTO(temp, entity);
		m_contents.splice(temp);
		if(m_contents.size() >= m_contentLength){
			print();
		}
	}
	void onChunkedTrailer(boost::uint64_t realContentLength, const OptionalMap &headers) OVERRIDE {
		LOG_POSEIDON_DEBUG("onChunkedTrailer(): realContentLength = ", realContentLength);
		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			LOG_POSEIDON_DEBUG("> ", it->first, " = ", Http::urlDecode(it->second));
		}
		print();
	}
	void onContentEof(boost::uint64_t realContentLength) OVERRIDE {
		LOG_POSEIDON_DEBUG("onContentEof(): realContentLength = ", realContentLength);
		print();
	}
};

MODULE_RAII(handles){
	handles.push(MyClient::create());
}

}
*/


#include "../src/precompiled.hpp"
#include "../src/async_job.hpp"
#include "../src/log.hpp"
#include "../src/time.hpp"
#include "../src/module_raii.hpp"

#include "../src/mysql/object_base.hpp"
#include "../src/singletons/mysql_daemon.hpp"

#define MYSQL_OBJECT_NAME	MySqlObj
#define MYSQL_OBJECT_FIELDS	\
	FIELD_SMALLINT(si)	\
	FIELD_STRING(str)	\
	FIELD_BIGINT(bi)	\
	FIELD_DATETIME(dt)
#include "../src/mysql/object_generator.hpp"

MODULE_RAII(/* handles */){
	using namespace Poseidon;

	AUTO(obj, boost::make_shared<MySqlObj>());
	obj->enableAutoSaving();
	obj->set_si(999);
	obj->set_str(std::string("\r\n\'\"\x00\x1A\\", 7));
	for(int i = 0; i < 10; ++i){
		obj->set_bi(i);
	}
//	obj->asyncLoad("SELECT * FROM `MySqlObj`");

	const auto now = getFastMonoClock();
	enqueueAsyncJob([]{ LOG_POSEIDON_FATAL("delayed 5000"); }, [=]{ return now + 5000 < getFastMonoClock(); });
	enqueueAsyncJob([]{ LOG_POSEIDON_FATAL("delayed 1000"); }, [=]{ return now + 1000 < getFastMonoClock(); });
	LOG_POSEIDON_FATAL("enqueued!");
}
