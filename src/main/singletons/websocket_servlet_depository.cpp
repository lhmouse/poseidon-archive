// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "websocket_servlet_depository.hpp"
#include <map>
#include <boost/ref.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/locks.hpp>
#include "main_config.hpp"
#include "../log.hpp"
#include "../exception.hpp"
using namespace Poseidon;

struct Poseidon::WebSocketServlet : NONCOPYABLE {
	const SharedNts uri;
	const boost::shared_ptr<WebSocketServletCallback> callback;

	WebSocketServlet(SharedNts uri_, boost::shared_ptr<WebSocketServletCallback> callback_)
		: uri(STD_MOVE(uri_), true), callback(STD_MOVE(callback_))
	{
		LOG_POSEIDON_INFO("Created WebSocket servlet for URI ", uri);
	}
	~WebSocketServlet(){
		LOG_POSEIDON_INFO("Destroyed WebSocket servlet for URI ", uri);
	}
};

namespace {

std::size_t g_maxRequestLength			= 16 * 0x400;
unsigned long long g_keepAliveTimeout   = 30000;

typedef std::map<std::size_t,
	std::map<SharedNts, boost::weak_ptr<WebSocketServlet> >
	> ServletMap;

boost::shared_mutex g_mutex;
ServletMap g_servlets;

}

void WebSocketServletDepository::start(){
	LOG_POSEIDON_INFO("Starting websocket servlet depository...");

	AUTO_REF(conf, MainConfig::getConfigFile());

	conf.get(g_maxRequestLength, "websocket_max_request_length");
	LOG_POSEIDON_DEBUG("Max request length = ", g_maxRequestLength);

	conf.get(g_keepAliveTimeout, "websocket_keep_alive_timeout");
	LOG_POSEIDON_DEBUG("Keep alive timeout = ", g_keepAliveTimeout);
}
void WebSocketServletDepository::stop(){
	LOG_POSEIDON_INFO("Unloading all WebSocket servlets...");

	ServletMap servlets;
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		servlets.swap(g_servlets);
	}
}

std::size_t WebSocketServletDepository::getMaxRequestLength(){
	return g_maxRequestLength;
}
unsigned long long WebSocketServletDepository::getKeepAliveTimeout(){
	return g_keepAliveTimeout;
}

boost::shared_ptr<WebSocketServlet> WebSocketServletDepository::registerServlet(
	std::size_t category, SharedNts uri, WebSocketServletCallback callback)
{
	AUTO(sharedCallback, boost::make_shared<WebSocketServletCallback>());
	sharedCallback->swap(callback);
	AUTO(servlet, boost::make_shared<WebSocketServlet>(uri, sharedCallback));
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		AUTO_REF(old, g_servlets[category][uri]);
		if(!old.expired()){
			LOG_POSEIDON_ERROR("Duplicate servlet for URI ", uri, " in category ", category);
			DEBUG_THROW(Exception, SharedNts::observe("Duplicate WebSocket servlet"));
		}
		old = servlet;
	}
	LOG_POSEIDON_DEBUG("Created servlet for for URI ", uri, " in category ", category);
	return servlet;
}

boost::shared_ptr<const WebSocketServletCallback> WebSocketServletDepository::getServlet(
	std::size_t category, const char *uri)
{
	if(!uri){
		LOG_POSEIDON_ERROR("uri is null");
		DEBUG_THROW(Exception, SharedNts::observe("uri is null"));
	}
	const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
	const AUTO(it, g_servlets.find(category));
	if(it == g_servlets.end()){
		LOG_POSEIDON_DEBUG("No servlet in category ", category);
		return VAL_INIT;
	}
	const AUTO(it2, it->second.find(SharedNts::observe(uri)));
	if(it2 == it->second.end()){
		LOG_POSEIDON_DEBUG("No servlet for URI ", uri, " in category ", category);
		return VAL_INIT;
	}
	const AUTO(servlet, it2->second.lock());
	if(!servlet){
		LOG_POSEIDON_DEBUG("Expired servlet for URI ", uri, " in category ", category);
		return VAL_INIT;
	}
	return servlet->callback;
}
