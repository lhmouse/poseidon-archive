#include "../precompiled.hpp"
#include "websocket_servlet_manager.hpp"
#include <map>
#include <boost/ref.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/locks.hpp>
#include "../log.hpp"
#include "../exception.hpp"
using namespace Poseidon;

struct Poseidon::WebSocketServlet : boost::noncopyable {
	const SharedNtmbs uri;
	const boost::shared_ptr<WebSocketServletCallback> callback;

	WebSocketServlet(SharedNtmbs uri_, boost::shared_ptr<WebSocketServletCallback> callback_)
		: uri(STD_MOVE(uri_), true), callback(STD_MOVE(callback_))
	{
		LOG_INFO("Created WebSocket servlet for URI ", uri);
	}
	~WebSocketServlet(){
		LOG_INFO("Destroyed WebSocket servlet for URI ", uri);
	}
};

namespace {

typedef std::map<std::size_t,
	std::map<SharedNtmbs, boost::weak_ptr<WebSocketServlet> >
	> ServletMap;

boost::shared_mutex g_mutex;
ServletMap g_servlets;

}

void WebSocketServletManager::start(){
}
void WebSocketServletManager::stop(){
	LOG_INFO("Unloading all WebSocket servlets...");

	ServletMap servlets;
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		servlets.swap(g_servlets);
	}
}

boost::shared_ptr<WebSocketServlet> WebSocketServletManager::registerServlet(
	std::size_t category, SharedNtmbs uri, WebSocketServletCallback callback)
{
	AUTO(sharedCallback, boost::make_shared<WebSocketServletCallback>());
	sharedCallback->swap(callback);
	uri.forkOwning();
	AUTO(servlet, boost::make_shared<WebSocketServlet>(uri, sharedCallback));
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		AUTO_REF(old, g_servlets[category][uri]);
		if(!old.expired()){
			LOG_ERROR("Duplicate WebSocket servlet: ", uri, " in category ", category);
			DEBUG_THROW(Exception, "Duplicate WebSocket servlet");
		}
		old = servlet;
	}
	return servlet;
}

boost::shared_ptr<const WebSocketServletCallback> WebSocketServletManager::getServlet(
	std::size_t category, const SharedNtmbs &uri)
{
	const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
	const AUTO(it, g_servlets.find(category));
	if(it == g_servlets.end()){
		LOG_DEBUG("No servlet in category ", category);
		return VAL_INIT;
	}
	const AUTO(it2, it->second.find(uri));
	if(it2 == it->second.end()){
		LOG_DEBUG("No servlet for URI ", uri, " in category ", category);
		return VAL_INIT;
	}
	const AUTO(servlet, it2->second.lock());
	if(!servlet){
		LOG_DEBUG("Servlet for URI ", uri, " in category ", category, " has expired");
		return VAL_INIT;
	}
	return servlet->callback;
}
