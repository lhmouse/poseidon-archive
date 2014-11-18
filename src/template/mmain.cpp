// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

// 这个文件被置于公有领域（public domain）。

#include "../main/precompiled.hpp"
#include "../main/log.hpp"
#include "../main/exception.hpp"
#include "../main/profiler.hpp"
#include "../main/hash.hpp"
#include "../main/module_raii.hpp"
#include "../main/singletons/http_servlet_manager.hpp"
#include "../main/singletons/module_manager.hpp"
#include "../main/singletons/event_listener_manager.hpp"
#include "../main/singletons/timer_daemon.hpp"
#include "../main/singletons/websocket_servlet_manager.hpp"
#include "../main/singletons/profile_manager.hpp"
#include "../main/http/websocket/session.hpp"
#include "../main/http/utilities.hpp"
#include "../main/tcp_client_base.hpp"
#include "../main/player/session.hpp"
#include "../main/http/session.hpp"
#include "../main/player/protocol_base.hpp"
#include "../main/singletons/player_servlet_manager.hpp"
#include "../main/mysql/object_base.hpp"
using namespace Poseidon;

#define MYSQL_OBJECT_NAME	MySqlObj
#define MYSQL_OBJECT_FIELDS	\
	FIELD_SMALLINT(si)	\
	FIELD_STRING(str)	\
	FIELD_BIGINT(bi)
#include "../main/mysql/object_generator.hpp"

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

void tickProc(unsigned long long now, unsigned long long period){
	PROFILE_ME;
	LOG_POSEIDON_FATAL("Tick, now = ", now, ", period = ", period);
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

	AUTO(obj, boost::make_shared<MySqlObj>());
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
		v.push_back(HttpServletManager::registerServlet(1, "/meow/", &meowProc));
		v.push_back(HttpServletManager::registerServlet(1, "/meow/meow/", &meowMeowProc));
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
	LOG_POSEIDON_FATAL("Received packet: opcode = ", opcode, ", payload = ", HexDumper(incoming));

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
		: TcpClientBase(IpPort("127.0.0.1", 443))
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
	TestInt req(incoming);
	LOG_POSEIDON_WARN("sint = ", req.i);
	req.i /= 10;
	ps->send(1000, req);
}
void TestUIntProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	TestUInt req(incoming);
	LOG_POSEIDON_WARN("int = ", req.u);
	req.u /= 10;
	ps->send(1001, req);
}
void TestStringProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	TestString req(incoming);
	LOG_POSEIDON_WARN("string = ", req.s);
	req.s += "_0123456789";
	ps->send(1002, req);
}

void TestIntArrayProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	TestIntArray req(incoming);
	LOG_POSEIDON_WARN("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARN("  ", i, " = ", req.a.at(i).i);
		req.a.at(i).i /= 10;
	}
	ps->send(1003, req);
}
void TestUIntArrayProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	TestUIntArray req(incoming);
	LOG_POSEIDON_WARN("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARN("  ", i, " = ", req.a.at(i).u);
		req.a.at(i).u /= 10;
	}
	ps->send(1004, req);
}
void TestStringArrayProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	TestStringArray req(incoming);
	LOG_POSEIDON_WARN("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARN("  ", i, " = ", req.a.at(i).s);
		req.a.at(i).s += "_0123456789";
	}
	ps->send(1005, req);
}

void TestProc(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming){
	LOG_POSEIDON_WARN("Received: ", HexDumper(incoming));
	TestProtocol req(incoming);
	LOG_POSEIDON_WARN("req.i = ", req.i);
	LOG_POSEIDON_WARN("req.j = ", req.j);
	LOG_POSEIDON_WARN("req.a.size() = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARN("req.a[", i, "].s = ", req.a.at(i).s);
		LOG_POSEIDON_WARN("req.a[", i, "].k = ", req.a.at(i).k);
	}
	ps->send(1006, req);
}

}

MODULE_RAII(
	return HttpServletManager::registerServlet(1, "/profile", &profileProc);
)
MODULE_RAII(
	return HttpServletManager::registerServlet(1, "/load", &loadProc);
)
MODULE_RAII(
	return HttpServletManager::registerServlet(1, "/unload", &unloadProc);
)
MODULE_RAII(
	AUTO(v, boost::make_shared<std::vector<boost::shared_ptr<void> > >());
	g_servlets = v;
	return v;
)
MODULE_RAII(
	return EventListenerManager::registerListener<TestEvent1>(&event1Proc);
)
MODULE_RAII(
	return EventListenerManager::registerListener<TestEvent2>(&event2Proc);
)
MODULE_RAII(
	return WebSocketServletManager::registerServlet(1, "/wstest", &webSocketProc);
)
MODULE_RAII(
	return PlayerServletManager::registerServlet(2, 100, &TestIntProc);
)
MODULE_RAII(
	return PlayerServletManager::registerServlet(2, 101, &TestUIntProc);
)
MODULE_RAII(
	return PlayerServletManager::registerServlet(2, 102, &TestStringProc);
)
MODULE_RAII(
	return PlayerServletManager::registerServlet(2, 103, &TestIntArrayProc);
)
MODULE_RAII(
	return PlayerServletManager::registerServlet(2, 104, &TestUIntArrayProc);
)
MODULE_RAII(
	return PlayerServletManager::registerServlet(2, 105, &TestStringArrayProc);
)
MODULE_RAII(
	return PlayerServletManager::registerServlet(2, 106, &TestProc);
)
MODULE_RAII(
	TestClient::create()->send(StreamBuffer("GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"));
)
