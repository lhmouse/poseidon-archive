// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "websocket_dispatcher_depository.hpp"
#include <boost/ref.hpp>
#include <boost/thread/mutex.hpp>
#include "main_config.hpp"
#include "../log.hpp"
#include "../exception.hpp"

namespace Poseidon {

struct WebSocketDispatcherDepository::Dispatcher : NONCOPYABLE {
	const SharedNts uri;
	const boost::shared_ptr<WebSocket::DispatcherCallback> callback;

	Dispatcher(SharedNts uri_, boost::shared_ptr<WebSocket::DispatcherCallback> callback_)
		: uri(STD_MOVE(uri_)), callback(STD_MOVE(callback_))
	{
		LOG_POSEIDON_INFO("Created WebSocket dispatcher for URI ", uri);
	}
	~Dispatcher(){
		LOG_POSEIDON_INFO("Destroyed WebSocket dispatcher for URI ", uri);
	}
};

namespace {
	std::size_t g_maxRequestLength		= 16 * 0x400;
	boost::uint64_t g_keepAliveTimeout	= 30000;

	typedef std::map<std::size_t,
		std::map<SharedNts, boost::weak_ptr<WebSocketDispatcherDepository::Dispatcher> >
		> DispatcherMap;

	boost::mutex g_mutex;
	DispatcherMap g_dispatchers;
}

void WebSocketDispatcherDepository::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting WebSocket dispatcher depository...");

	AUTO_REF(conf, MainConfig::getConfigFile());

	conf.get(g_maxRequestLength, "_max_request_length");
	LOG_POSEIDON_DEBUG("Max request length = ", g_maxRequestLength);

	conf.get(g_keepAliveTimeout, "_keep_alive_timeout");
	LOG_POSEIDON_DEBUG("Keep alive timeout = ", g_keepAliveTimeout);
}
void WebSocketDispatcherDepository::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unloading all WebSocket dispatchers...");

	DispatcherMap dispatchers;
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		dispatchers.swap(g_dispatchers);
	}
}

std::size_t WebSocketDispatcherDepository::getMaxRequestLength(){
	return g_maxRequestLength;
}
boost::uint64_t WebSocketDispatcherDepository::getKeepAliveTimeout(){
	return g_keepAliveTimeout;
}

boost::shared_ptr<WebSocketDispatcherDepository::Dispatcher> WebSocketDispatcherDepository::create(
	std::size_t category, SharedNts uri, WebSocket::DispatcherCallback callback)
{
	AUTO(sharedCallback, boost::make_shared<WebSocket::DispatcherCallback>());
	sharedCallback->swap(callback);
	AUTO(dispatcher, boost::make_shared<Dispatcher>(uri, sharedCallback));
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		AUTO_REF(old, g_dispatchers[category][uri]);
		if(!old.expired()){
			LOG_POSEIDON_ERROR("Duplicate dispatcher for URI ", uri, " in category ", category);
			DEBUG_THROW(Exception, SharedNts::observe("Duplicate WebSocket dispatcher"));
		}
		old = dispatcher;
	}
	LOG_POSEIDON_DEBUG("Created dispatcher for for URI ", uri, " in category ", category);
	return dispatcher;
}

boost::shared_ptr<const WebSocket::DispatcherCallback> WebSocketDispatcherDepository::get(std::size_t category, const char *uri){
	if(!uri){
		LOG_POSEIDON_ERROR("URI is null");
		DEBUG_THROW(Exception, SharedNts::observe("URI is null"));
	}
	const boost::mutex::scoped_lock lock(g_mutex);
	const AUTO(it, g_dispatchers.find(category));
	if(it == g_dispatchers.end()){
		LOG_POSEIDON_DEBUG("No dispatcher in category ", category);
		return VAL_INIT;
	}
	const AUTO(it2, it->second.find(SharedNts::observe(uri)));
	if(it2 == it->second.end()){
		LOG_POSEIDON_DEBUG("No dispatcher for URI ", uri, " in category ", category);
		return VAL_INIT;
	}
	const AUTO(dispatcher, it2->second.lock());
	if(!dispatcher){
		LOG_POSEIDON_DEBUG("Expired dispatcher for URI ", uri, " in category ", category);
		return VAL_INIT;
	}
	return dispatcher->callback;
}

}
