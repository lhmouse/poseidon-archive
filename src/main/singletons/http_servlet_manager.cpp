#include "../precompiled.hpp"
#include "http_servlet_manager.hpp"
#include <string>
#include <map>
#include <boost/ref.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "main_config.hpp"
#include "../log.hpp"
#include "../exception.hpp"
using namespace Poseidon;

struct Poseidon::HttpServlet : boost::noncopyable {
	const SharedNtmbs uri;
	const boost::shared_ptr<const HttpServletCallback> callback;

	HttpServlet(SharedNtmbs uri_, boost::shared_ptr<const HttpServletCallback> callback_)
		: uri(STD_MOVE(uri_)), callback(STD_MOVE(callback_))
	{
		LOG_INFO("Created HTTP servlet for URI ", uri);
	}
	~HttpServlet(){
		LOG_INFO("Destroyed HTTP servlet for URI ", uri);
	}
};

namespace {

std::size_t g_maxRequestLength = 16 * 0x400;
unsigned long long g_requestTimeout = 30000;
unsigned long long g_keepAliveTimeout = 5000;

typedef std::map<unsigned,
	std::map<SharedNtmbs, boost::weak_ptr<HttpServlet> >
	> ServletMap;

boost::shared_mutex g_mutex;
ServletMap g_servlets;

bool getExactServlet(boost::shared_ptr<const HttpServletCallback> &ret,
	unsigned port, const char *uri, std::size_t uriLen)
{
	const boost::shared_lock<boost::shared_mutex> slock(g_mutex);

	const AUTO(it, g_servlets.find(port));
	if(it == g_servlets.end()){
		LOG_DEBUG("No servlet on port ", port);
		return false;
	}

	AUTO_REF(servletsOnPort, it->second);
	const AUTO(sit, servletsOnPort.lower_bound(uri));
	if(sit == servletsOnPort.end()){
		LOG_DEBUG("No more handlers: ", uri);
		return false;
	}
	const std::size_t len = std::strlen(sit->first.get());
	if((len < uriLen) || (std::memcmp(sit->first.get(), uri, uriLen) != 0)){
		LOG_DEBUG("No more handlers: ", uri);
		return false;
	}
	if(len == uriLen){
		const AUTO(servlet, sit->second.lock());
		if(servlet){
			ret = servlet->callback;
		}
	}
	return true;
}

}

void HttpServletManager::start(){
	MainConfig::get(g_maxRequestLength, "http_max_request_length");
	LOG_DEBUG("Max request length = ", g_maxRequestLength);

	MainConfig::get(g_requestTimeout, "http_request_timeout");
	LOG_DEBUG("Request timeout = ", g_requestTimeout);

	MainConfig::get(g_keepAliveTimeout, "http_keep_alive_timeout");
	LOG_DEBUG("Keep-Alive timeout = ", g_keepAliveTimeout);
}
void HttpServletManager::stop(){
	LOG_INFO("Unloading all HTTP servlets...");

	ServletMap servlets;
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		servlets.swap(g_servlets);
	}
}

std::size_t HttpServletManager::getMaxRequestLength(){
	return g_maxRequestLength;
}
unsigned long long HttpServletManager::getRequestTimeout(){
	return g_requestTimeout;
}
unsigned long long HttpServletManager::getKeepAliveTimeout(){
	return g_keepAliveTimeout;
}

boost::shared_ptr<HttpServlet> HttpServletManager::registerServlet(
	unsigned port, SharedNtmbs uri, HttpServletCallback callback)
{
	AUTO(sharedCallback, boost::make_shared<HttpServletCallback>());
	sharedCallback->swap(callback);
	uri.forkOwning();
	AUTO(servlet, boost::make_shared<HttpServlet>(uri, sharedCallback));
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		AUTO_REF(old, g_servlets[port][uri]);
		if(!old.expired()){
			LOG_ERROR("Duplicate HTTP servlet for URI ", uri, " on port ", port);
			DEBUG_THROW(Exception, "Duplicate HTTP servlet");
		}
		old = servlet;
	}
	return servlet;
}

boost::shared_ptr<const HttpServletCallback> HttpServletManager::getServlet(
	unsigned port, const SharedNtmbs &uri)
{
	if(uri[0] != '/'){
		LOG_ERROR("URI must begin with a slash: ", uri);
		DEBUG_THROW(Exception, "Bad URI format");
	}

	boost::shared_ptr<const HttpServletCallback> ret;
	const std::size_t uriLen = std::strlen(uri.get());
	getExactServlet(ret, port, uri.get(), uriLen);
	if(!ret && (uri != "/")){
		LOG_DEBUG("Searching for fallback handlers for URI ", uri);

		std::string fallback;
		fallback.reserve(uriLen + 1);
		fallback.push_back('/');
		std::size_t slash = 0;
		for(;;){
			LOG_DEBUG("Trying fallback URI handler ", fallback);

			boost::shared_ptr<const HttpServletCallback> test;
			if(!getExactServlet(test, port, fallback.c_str(), fallback.size())){
				break;
			}
			if(test){
				LOG_DEBUG("Fallback handler matches: ", fallback);
				ret.swap(test);
			}

			if(slash >= uriLen - 1){
				break;
			}
			const char *const nextSlash = std::strchr(uri.get() + slash + 1, '/');
			if(nextSlash){
				fallback.append(uri.get() + slash + 1, nextSlash);
			} else {
				fallback.append(uri.get() + slash + 1);
			}
			fallback.push_back('/');
			slash = nextSlash - uri.get();
		}
	}
	return ret;
}
