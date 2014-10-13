#include "../precompiled.hpp"
#include "../main/module.hpp"
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
using namespace Poseidon;

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
	LOG_FATAL("event1Proc: i = ", event->i, ", s = ", event->s);
}

void event2Proc(boost::shared_ptr<TestEvent2> event){
	PROFILE_ME;
	LOG_FATAL("event2Proc: d = ", event->d);
}

boost::shared_ptr<const HttpServlet> g_load, g_unload, g_meow, g_meowMeow;
boost::shared_ptr<const EventListener> g_event1, g_event2;
boost::shared_ptr<const TimerItem> g_tick;
boost::shared_ptr<const WebSocketServlet> g_ws;

void tickProc(unsigned long long now, unsigned long long period){
	PROFILE_ME;
	LOG_FATAL("Tick, now = ", now, ", period = ", period);
}

HttpStatus meowProc(OptionalMap &headers, StreamBuffer &contents, HttpRequest){
	PROFILE_ME;

	AUTO(event, boost::make_shared<TestEvent1>());
	event->i = 123;
	event->s = "meow";
	event->raise();

	headers.set("Content-Type", "text/html");
	contents.put("<h1>Meow!</h1>");
	return HTTP_OK;
}
HttpStatus meowMeowProc(OptionalMap &headers, StreamBuffer &contents, HttpRequest){
	PROFILE_ME;

	AUTO(event, boost::make_shared<TestEvent1>());
	event->i = 123;
	event->s = "meow/meow";
	event->raise();

	headers.set("Content-Type", "text/html");
	contents.put("<h1>Meow! Meow!</h1>");
	return HTTP_OK;
}
HttpStatus loadProc(OptionalMap &, StreamBuffer &contents, HttpRequest,
	const boost::weak_ptr<const Module> &module)
{
	PROFILE_ME;

	AUTO(event, boost::make_shared<TestEvent2>());
	event->d = 123.45;
	event->raise();

	if(g_meow){
		contents.put("Already loaded");
		return HTTP_OK;
	}
	// 通配路径 /meow/*
	g_meow = HttpServletManager::registerServlet("/meow/", module, &meowProc);
	g_meowMeow = HttpServletManager::registerServlet("/meow/meow/", module, &meowMeowProc);
	g_tick = TimerDaemon::registerTimer(5000, 10000, module, &tickProc);
	contents.put("OK");
	return HTTP_OK;
}
HttpStatus unloadProc(OptionalMap &, StreamBuffer &contents, HttpRequest request){
	PROFILE_ME;

	AUTO(event, boost::make_shared<TestEvent2>());
	event->d = 67.89;
	event->raise();

	if(request.getParams["unload_module"] == "1"){
		ModuleManager::unload("libposeidon-template.so");
		contents.put("Module unloaded");
		return HTTP_OK;
	}
	if(!g_meow){
		contents.put("Already unloaded");
		return HTTP_OK;
	}
	g_meow.reset();
	g_meowMeow.reset();
	g_tick.reset();
	contents.put("OK");
	return HTTP_OK;
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

struct Tracked {
	Tracked(){
		LOG_FATAL("Tracked::Tracked()");
	}
	~Tracked(){
		LOG_FATAL("Tracked::~Tracked()");
	}
} g_tracked;

}

extern "C" void poseidonModuleInit(const boost::weak_ptr<const Module> &module){
	LOG_FATAL("poseidonModuleInit()");

	using namespace TR1::placeholders;
	g_load = HttpServletManager::registerServlet("/load", module,
		TR1::bind(&loadProc, _1, _2, _3, module));
	g_unload = HttpServletManager::registerServlet("/unload", module, &unloadProc);

	g_event1 = EventListenerManager::registerListener<TestEvent1>(module, &event1Proc);
	g_event2 = EventListenerManager::registerListener<TestEvent2>(module, &event2Proc);

	g_ws = WebSocketServletManager::registerServlet("/wstest", module, &webSocketProc);
}
