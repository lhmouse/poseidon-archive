#include "../main/precompiled.hpp"
#include "../main/singletons/module_manager.hpp"
#include "../main/log.hpp"
#include "../main/exception.hpp"
#include "../main/singletons/http_servlet_manager.hpp"
#include "../main/singletons/module_manager.hpp"
#include "../main/singletons/event_listener_manager.hpp"
#include "../main/singletons/timer_daemon.hpp"
#include "../main/singletons/websocket_servlet_manager.hpp"
#include "../main/http/websocket/session.hpp"
#include "../main/http/utilities.hpp"
#include "../main/hash.hpp"
#include "../main/profiler.hpp"
#include "../main/singletons/profile_manager.hpp"
#include "../main/tcp_client_base.hpp"
#include "../main/player/session.hpp"
#include "../main/http/session.hpp"
#include "../main/player/protocol_base.hpp"
#include "../main/singletons/player_servlet_manager.hpp"
#include "../main/mysql/object_base.hpp"
#include "../main/uuid.hpp"
using namespace Poseidon;

#define MYSQL_OBJECT_NAMESPACE	TestNs

#define MYSQL_OBJECT_NAME	MySqlObj
#define MYSQL_OBJECT_FIELDS	\
	FIELD_SMALLINT(si)	\
	FIELD_STRING(str)	\
	FIELD_BIGINT(bi)
#include "../main/mysql/object_generator.hpp"

namespace {

struct Tracked {
	Tracked(){
 		LOG_FATAL("Tracked::Tracked()");
	}
	~Tracked(){
		LOG_FATAL("Tracked::~Tracked()");
	}
};

struct TestEvent1 : public EventBase<1> {
	int i;
	std::string s;
};

struct TestEvent2 : public EventBase<1> {
	double d;
};

void event1Proc(boost::shared_ptr<TestEvent1> event){
	PROFILE_ME;
	LOG_FATAL("event1Proc: i = ", event->i, ", s = ", event->s);
}

void event2Proc(boost::shared_ptr<TestEvent2> event){
	PROFILE_ME;
	LOG_FATAL("event2Proc: d = ", event->d);
}

boost::shared_ptr<const HttpServlet> g_profile, g_load, g_unload, g_meow, g_meowMeow;
boost::shared_ptr<const EventListener> g_event1, g_event2;
boost::shared_ptr<const TimerItem> g_tick;
boost::shared_ptr<const WebSocketServlet> g_ws;
std::vector<boost::shared_ptr<const PlayerServlet> > g_player;

void tickProc(unsigned long long now, unsigned long long period){
	PROFILE_ME;
	LOG_FATAL("Tick, now = ", now, ", period = ", period);
}

void profileProc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	OptionalMap headers;
	StreamBuffer contents;

	headers.set("Content-Type", "text/plain");
	contents.put("   Samples      Total time(us)  Exclusive time(us)    File:Line\n");
	const AUTO(profile, ProfileManager::snapshot());
	for(AUTO(it, profile.begin()); it != profile.end(); ++it){
		char temp[128];
		int len = std::sprintf(temp, "%10llu%20llu%20llu    ",
			it->samples, it->usTotal, it->usExclusive);
		contents.put(temp, len);
		contents.put(it->file.get());
		len = std::sprintf(temp, ":%lu\n", it->line);
		contents.put(temp, len);
	}
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

void meowProc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	AUTO(obj, boost::make_shared<TestNs::MySqlObj>());
	obj->set_si(123);
	obj->set_str("meow");
	obj->set_bi(456789);
	obj->asyncSave();

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
void loadProc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	OptionalMap headers;
	StreamBuffer contents;

	AUTO(event, boost::make_shared<TestEvent2>());
	event->d = 123.45;
	event->raise();

	if(g_meow){
		contents.put("Already loaded");
	} else {
		// 通配路径 /meow/ *
		g_meow = HttpServletManager::registerServlet(8860, "/meow/", &meowProc);
		g_meowMeow = HttpServletManager::registerServlet(8860, "/meow/meow/", &meowMeowProc);
		g_tick = TimerDaemon::registerTimer(5000, 10000, &tickProc);
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

	if(!g_meow){
		contents.put("Already unloaded");
	} else {
		g_meow.reset();
		g_meowMeow.reset();
		g_tick.reset();
		contents.put("OK");
	}
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

void webSocketProc(boost::shared_ptr<WebSocketSession> wss,
	WebSocketOpCode opcode, StreamBuffer incoming)
{
	PROFILE_ME;
	LOG_FATAL("Received packet: opcode = ", opcode, ", payload = ", HexDumper(incoming));

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
		ret->sslConnect();
		ret->goResident();
		return ret;
	}

public:
	TestClient()
		: TcpClientBase("127.0.0.1", 443)
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

#define PROTOCOL_NAMESPACE TestNs

#define PROTOCOL_NAME		TestInt
#define PROTOCOL_ID			100
#define PROTOCOL_FIELDS		FIELD_VINT(i)
#include "../main/player/protocol_generator.hpp"

#define PROTOCOL_NAME		TestUInt
#define PROTOCOL_ID			101
#define PROTOCOL_FIELDS 	FIELD_VUINT(u)
#include "../main/player/protocol_generator.hpp"

#define PROTOCOL_NAME		TestString
#define PROTOCOL_ID			102
#define PROTOCOL_FIELDS		FIELD_STRING(s)
#include "../main/player/protocol_generator.hpp"

#define PROTOCOL_NAME		TestIntArray
#define PROTOCOL_ID			103
#define PROTOCOL_FIELDS		FIELD_ARRAY(a, FIELD_VINT(i))
#include "../main/player/protocol_generator.hpp"

#define PROTOCOL_NAME		TestUIntArray
#define PROTOCOL_ID			104
#define PROTOCOL_FIELDS		FIELD_ARRAY(a, FIELD_VUINT(u))
#include "../main/player/protocol_generator.hpp"

#define PROTOCOL_NAME		TestStringArray
#define PROTOCOL_ID			105
#define PROTOCOL_FIELDS		FIELD_ARRAY(a, FIELD_STRING(s))
#include "../main/player/protocol_generator.hpp"

#define PROTOCOL_NAME   	TestProtocol
#define PROTOCOL_ID			106
#define PROTOCOL_FIELDS \
	FIELD_VINT(i)   \
	FIELD_VUINT(j)  \
	FIELD_ARRAY(a,  \
		FIELD_STRING(s) \
		FIELD_VUINT(k)  \
	)
#include "../main/player/protocol_generator.hpp"

namespace {

void TestIntProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	TestNs::TestInt req(incoming);
	LOG_WARN("sint = ", req.i);
	req.i /= 10;
	ps->send(1000, req);
}
void TestUIntProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	TestNs::TestUInt req(incoming);
	LOG_WARN("int = ", req.u);
	req.u /= 10;
	ps->send(1001, req);
}
void TestStringProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	TestNs::TestString req(incoming);
	LOG_WARN("string = ", req.s);
	req.s += "_0123456789";
	ps->send(1002, req);
}

void TestIntArrayProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	TestNs::TestIntArray req(incoming);
	LOG_WARN("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_WARN("  ", i, " = ", req.a.at(i).i);
		req.a.at(i).i /= 10;
	}
	ps->send(1003, req);
}
void TestUIntArrayProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	TestNs::TestUIntArray req(incoming);
	LOG_WARN("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_WARN("  ", i, " = ", req.a.at(i).u);
		req.a.at(i).u /= 10;
	}
	ps->send(1004, req);
}
void TestStringArrayProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	TestNs::TestStringArray req(incoming);
	LOG_WARN("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_WARN("  ", i, " = ", req.a.at(i).s);
		req.a.at(i).s += "_0123456789";
	}
	ps->send(1005, req);
}

void TestProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	LOG_WARN("Received: ", HexDumper(incoming));
	TestNs::TestProtocol req(incoming);
	LOG_WARN("req.i = ", req.i);
	LOG_WARN("req.j = ", req.j);
	LOG_WARN("req.a.size() = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_WARN("req.a[", i, "].s = ", req.a.at(i).s);
		LOG_WARN("req.a[", i, "].k = ", req.a.at(i).k);
	}
	ps->send(1006, req);
}

}

extern "C" void poseidonModuleInit(std::vector<boost::shared_ptr<const void> > &contexts){
	LOG_FATAL("poseidonModuleInit()");

	contexts.push_back(boost::make_shared<Tracked>());

	g_profile = HttpServletManager::registerServlet(8860, "/profile", &profileProc);

	g_load = HttpServletManager::registerServlet(8860, "/load", &loadProc);
	g_unload = HttpServletManager::registerServlet(8860, "/unload", &unloadProc);

	g_event1 = EventListenerManager::registerListener<TestEvent1>(&event1Proc);
	g_event2 = EventListenerManager::registerListener<TestEvent2>(&event2Proc);

	g_ws = WebSocketServletManager::registerServlet(8860, "/wstest", &webSocketProc);

	g_player.push_back(PlayerServletManager::registerServlet(8850, 100, &TestIntProc));
	g_player.push_back(PlayerServletManager::registerServlet(8850, 101, &TestUIntProc));
	g_player.push_back(PlayerServletManager::registerServlet(8850, 102, &TestStringProc));
	g_player.push_back(PlayerServletManager::registerServlet(8850, 103, &TestIntArrayProc));
	g_player.push_back(PlayerServletManager::registerServlet(8850, 104, &TestUIntArrayProc));
	g_player.push_back(PlayerServletManager::registerServlet(8850, 105, &TestStringArrayProc));
	g_player.push_back(PlayerServletManager::registerServlet(8850, 106, &TestProc));

	Uuid uuid = Uuid::createRandom();
	std::string str = uuid.toHex();
	LOG_DEBUG("UUID = ", str);
	Uuid uuid2 = Uuid::createFromString(str);
	LOG_DEBUG("UUID = ", uuid2.toHex());

	TestClient::create()->send(StreamBuffer("GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"));
}
