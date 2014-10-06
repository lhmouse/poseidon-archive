#include "../precompiled.hpp"
#include "../main/module.hpp"
#include "../main/log.hpp"
#include "../main/exception.hpp"
#include "../main/singletons/http_servlet_manager.hpp"
#include "../main/singletons/module_manager.hpp"
#include "../main/singletons/event_listener_manager.hpp"
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
	LOG_FATAL("event1Proc: i = ", event->i, ", s = ", event->s);
}

void event2Proc(boost::shared_ptr<TestEvent2> event){
	LOG_FATAL("event2Proc: d = ", event->d);
}

boost::shared_ptr<const HttpServlet> g_load, g_unload, g_meow;
boost::shared_ptr<const EventListener> g_event1, g_event2;

HttpStatus meowProc(OptionalMap &headers, StreamBuffer &contents, HttpRequest){
	AUTO(event, boost::make_shared<TestEvent1>());
	event->i = 123;
	event->s = "meow";
	event->raise();

	headers.set("Content-Type", "text/html");
	contents.put("<h1>Meow!</h1>");
	return HTTP_OK;
}
HttpStatus loadProc(OptionalMap &, StreamBuffer &contents, HttpRequest,
	const boost::weak_ptr<const Module> &module)
{
	AUTO(event, boost::make_shared<TestEvent2>());
	event->d = 123.45;
	event->raise();

	if(g_meow){
		contents.put("Already loaded");
		return HTTP_OK;
	}
	// 通配路径 /meow/*
	g_meow = HttpServletManager::registerServlet("/meow/", module, &meowProc);
	contents.put("OK");
	return HTTP_OK;
}
HttpStatus unloadProc(OptionalMap &, StreamBuffer &contents, HttpRequest request){
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
	contents.put("OK");
	return HTTP_OK;
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
}
