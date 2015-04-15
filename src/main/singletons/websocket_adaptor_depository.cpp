// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "websocket_adaptor_depository.hpp"
#include <boost/ref.hpp>
#include <boost/thread/mutex.hpp>
#include "main_config.hpp"
#include "../log.hpp"
#include "../exception.hpp"

namespace Poseidon {

struct WebSocketAdaptorDepository::Adaptor : NONCOPYABLE {
	const SharedNts uri;
	const boost::shared_ptr<WebSocket::AdaptorCallback> callback;

	Adaptor(SharedNts uri_, boost::shared_ptr<WebSocket::AdaptorCallback> callback_)
		: uri(STD_MOVE(uri_)), callback(STD_MOVE(callback_))
	{
		LOG_POSEIDON_INFO("Created WebSocket adaptor for URI ", uri);
	}
	~Adaptor(){
		LOG_POSEIDON_INFO("Destroyed WebSocket adaptor for URI ", uri);
	}
};

namespace {
	std::size_t g_maxRequestLength		= 16 * 0x400;
	boost::uint64_t g_keepAliveTimeout	= 30000;

	typedef std::map<std::size_t,
		std::map<SharedNts, boost::weak_ptr<WebSocketAdaptorDepository::Adaptor> >
		> AdaptorMap;

	boost::mutex g_mutex;
	AdaptorMap g_adaptors;
}

void WebSocketAdaptorDepository::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting WebSocket adaptor depository...");

	AUTO_REF(conf, MainConfig::getConfigFile());

	conf.get(g_maxRequestLength, "_max_request_length");
	LOG_POSEIDON_DEBUG("Max request length = ", g_maxRequestLength);

	conf.get(g_keepAliveTimeout, "_keep_alive_timeout");
	LOG_POSEIDON_DEBUG("Keep alive timeout = ", g_keepAliveTimeout);
}
void WebSocketAdaptorDepository::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unloading all WebSocket adaptors...");

	AdaptorMap adaptors;
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		adaptors.swap(g_adaptors);
	}
}

std::size_t WebSocketAdaptorDepository::getMaxRequestLength(){
	return g_maxRequestLength;
}
boost::uint64_t WebSocketAdaptorDepository::getKeepAliveTimeout(){
	return g_keepAliveTimeout;
}

boost::shared_ptr<WebSocketAdaptorDepository::Adaptor> WebSocketAdaptorDepository::create(
	std::size_t category, SharedNts uri, WebSocket::AdaptorCallback callback)
{
	AUTO(sharedCallback, boost::make_shared<WebSocket::AdaptorCallback>());
	sharedCallback->swap(callback);
	AUTO(adaptor, boost::make_shared<Adaptor>(uri, sharedCallback));
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		AUTO_REF(old, g_adaptors[category][uri]);
		if(!old.expired()){
			LOG_POSEIDON_ERROR("Duplicate adaptor for URI ", uri, " in category ", category);
			DEBUG_THROW(Exception, SharedNts::observe("Duplicate WebSocket adaptor"));
		}
		old = adaptor;
	}
	LOG_POSEIDON_DEBUG("Created adaptor for for URI ", uri, " in category ", category);
	return adaptor;
}

boost::shared_ptr<const WebSocket::AdaptorCallback> WebSocketAdaptorDepository::get(std::size_t category, const char *uri){
	if(!uri){
		LOG_POSEIDON_ERROR("URI is null");
		DEBUG_THROW(Exception, SharedNts::observe("URI is null"));
	}
	const boost::mutex::scoped_lock lock(g_mutex);
	const AUTO(it, g_adaptors.find(category));
	if(it == g_adaptors.end()){
		LOG_POSEIDON_DEBUG("No adaptor in category ", category);
		return VAL_INIT;
	}
	const AUTO(it2, it->second.find(SharedNts::observe(uri)));
	if(it2 == it->second.end()){
		LOG_POSEIDON_DEBUG("No adaptor for URI ", uri, " in category ", category);
		return VAL_INIT;
	}
	const AUTO(adaptor, it2->second.lock());
	if(!adaptor){
		LOG_POSEIDON_DEBUG("Expired adaptor for URI ", uri, " in category ", category);
		return VAL_INIT;
	}
	return adaptor->callback;
}

}
