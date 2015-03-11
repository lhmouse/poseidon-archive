// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "websocket_servlet_depository.hpp"
#include <map>
#include <boost/ref.hpp>
#include <boost/thread/mutex.hpp>
#include "main_config.hpp"
#include "../log.hpp"
#include "../exception.hpp"

namespace Poseidon {

struct WebSocketServletDepository::Servlet : NONCOPYABLE {
	const SharedNts uri;
	const boost::shared_ptr<WebSocket::ServletCallback> callback;

	Servlet(SharedNts uri_, boost::shared_ptr<WebSocket::ServletCallback> callback_)
		: uri(STD_MOVE(uri_)), callback(STD_MOVE(callback_))
	{
		LOG_POSEIDON_INFO("Created WebSocket servlet for URI ", uri);
	}
	~Servlet(){
		LOG_POSEIDON_INFO("Destroyed WebSocket servlet for URI ", uri);
	}
};

namespace {
	std::size_t g_maxRequestLength		= 16 * 0x400;
	boost::uint64_t g_keepAliveTimeout	= 30000;

	typedef std::map<std::size_t,
		std::map<SharedNts, boost::weak_ptr<WebSocketServletDepository::Servlet> >
		> ServletMap;

	boost::mutex g_mutex;
	ServletMap g_servlets;
}

void WebSocketServletDepository::start(){
	LOG_POSEIDON_INFO("Starting WebSocket servlet depository...");

	AUTO_REF(conf, MainConfig::getConfigFile());

	conf.get(g_maxRequestLength, "_max_request_length");
	LOG_POSEIDON_DEBUG("Max request length = ", g_maxRequestLength);

	conf.get(g_keepAliveTimeout, "_keep_alive_timeout");
	LOG_POSEIDON_DEBUG("Keep alive timeout = ", g_keepAliveTimeout);
}
void WebSocketServletDepository::stop(){
	LOG_POSEIDON_INFO("Unloading all WebSocket servlets...");

	ServletMap servlets;
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		servlets.swap(g_servlets);
	}
}

std::size_t WebSocketServletDepository::getMaxRequestLength(){
	return g_maxRequestLength;
}
boost::uint64_t WebSocketServletDepository::getKeepAliveTimeout(){
	return g_keepAliveTimeout;
}

boost::shared_ptr<WebSocketServletDepository::Servlet> WebSocketServletDepository::create(
	std::size_t category, SharedNts uri, WebSocket::ServletCallback callback)
{
	AUTO(sharedCallback, boost::make_shared<WebSocket::ServletCallback>());
	sharedCallback->swap(callback);
	AUTO(servlet, boost::make_shared<Servlet>(uri, sharedCallback));
	{
		const boost::mutex::scoped_lock lock(g_mutex);
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

boost::shared_ptr<const WebSocket::ServletCallback> WebSocketServletDepository::get(std::size_t category, const char *uri){
	if(!uri){
		LOG_POSEIDON_ERROR("uri is null");
		DEBUG_THROW(Exception, SharedNts::observe("uri is null"));
	}
	const boost::mutex::scoped_lock lock(g_mutex);
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

}
