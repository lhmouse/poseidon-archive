#include "../../precompiled.hpp"
#include "http_servlet_manager.hpp"
#include <string>
#include <map>
#include <boost/noncopyable.hpp>
#include <boost/ref.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "config_file.hpp"
#include "../log.hpp"
#include "../exception.hpp"
using namespace Poseidon;

namespace {

class RealHttpServlet : boost::noncopyable,
	public boost::enable_shared_from_this<RealHttpServlet>
{
private:
	const std::string m_uri;
	const boost::weak_ptr<const void> m_dependency;
	const HttpServletCallback m_callback;

public:
	RealHttpServlet(const std::string &uri,
		const boost::weak_ptr<const void> &dependency, const HttpServletCallback &callback)
		: m_uri(uri), m_dependency(dependency), m_callback(callback)
	{
		LOG_INFO("Created HTTP servlet for URI ", m_uri);
	}
	~RealHttpServlet(){
		LOG_INFO("Destroyed HTTP servlet for URI ", m_uri);
	}

public:
	boost::shared_ptr<const HttpServletCallback> lock(boost::shared_ptr<const void> &lockedDep) const {
		if((m_dependency < boost::weak_ptr<void>()) || (boost::weak_ptr<void>() < m_dependency)){
			lockedDep = m_dependency.lock();
			if(!lockedDep){
				return VALUE_INIT;
			}
		}
		return boost::shared_ptr<const HttpServletCallback>(shared_from_this(), &m_callback);
	}
};

}

struct Poseidon::HttpServlet : boost::noncopyable {
	const boost::shared_ptr<RealHttpServlet> realHttpServlet;

	explicit HttpServlet(boost::shared_ptr<RealHttpServlet> realHttpServlet_)
		: realHttpServlet(STD_MOVE(realHttpServlet_))
	{
	}
};

namespace {

std::size_t g_maxRequestLength = 16 * 0x400;
unsigned long long g_requestTimeout = 30000;
unsigned long long g_keepAliveTimeout = 5000;

boost::shared_mutex g_mutex;
std::map<std::string, boost::weak_ptr<const HttpServlet> > g_servlets;

bool getExactServlet(boost::shared_ptr<const HttpServletCallback> &ret,
	boost::shared_ptr<const void> &lockedDep, const std::string &uri)
{
	const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
	const AUTO(it, g_servlets.lower_bound(uri));
	if((it == g_servlets.end()) || (it->first.size() < uri.size()) ||
		(it->first.compare(0, uri.size(), uri) != 0))
	{
		LOG_DEBUG("No more handlers: ", uri);
		return false;
	}
	if(it->first.size() == uri.size()){
		const AUTO(servlet, it->second.lock());
		if(servlet){
			servlet->realHttpServlet->lock(lockedDep).swap(ret);
		}
	}
	return true;
}

}

void HttpServletManager::start(){
	ConfigFile::get(g_maxRequestLength, "http_max_request_length");
	LOG_DEBUG("Max request length = ", g_maxRequestLength);

	ConfigFile::get(g_requestTimeout, "http_request_timeout");
	LOG_DEBUG("Request timeout = ", g_requestTimeout);

	ConfigFile::get(g_keepAliveTimeout, "http_keep_alive_timeout");
	LOG_DEBUG("Keep-Alive timeout = ", g_keepAliveTimeout);
}
void HttpServletManager::stop(){
	LOG_INFO("Unloading all HTTP servlets...");

	g_servlets.clear();
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

boost::shared_ptr<HttpServlet> HttpServletManager::registerServlet(const std::string &uri,
	const boost::weak_ptr<const void> &dependency, const HttpServletCallback &callback)
{
	AUTO(newServlet, boost::make_shared<HttpServlet>(boost::make_shared<RealHttpServlet>(
		boost::ref(uri), boost::ref(dependency), boost::ref(callback))));
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		AUTO_REF(servlet, g_servlets[uri]);
		if(!servlet.expired()){
			DEBUG_THROW(Exception, "Duplicate HTTP servlet.");
		}
		servlet = newServlet;
	}
	return newServlet;
}

boost::shared_ptr<const HttpServletCallback> HttpServletManager::getServlet(
	boost::shared_ptr<const void> &lockedDep, const std::string &uri)
{
	boost::shared_ptr<const HttpServletCallback> ret;
	if(uri[0] != '/'){
		LOG_DEBUG("URI must begin with /: ", uri);
		return ret;
	}

	getExactServlet(ret, lockedDep, uri);
	if(ret){
		return ret;
	}

	if(uri == "/"){
		return ret;
	}
	LOG_DEBUG("Searching for fallback handlers for URI ", uri);

	std::string fallback;
	fallback.reserve(uri.size() + 1);
	fallback.push_back('/');
	std::size_t slash = 0;
	for(;;){
		LOG_DEBUG("Trying fallback URI handler ", fallback);

		boost::shared_ptr<const HttpServletCallback> test;
		boost::shared_ptr<const void> testDep;
		if(!getExactServlet(test, testDep, fallback)){
			break;
		}
		if(test){
			LOG_DEBUG("Fallback handler matches: ", fallback);

			ret.swap(test);
			lockedDep.swap(testDep);
		}

		if(slash >= uri.size() - 1){
			break;
		}
		const std::size_t nextSlash = uri.find('/', slash + 1);
		if(nextSlash == std::string::npos){
			fallback.append(uri.begin() + slash + 1, uri.end());
		} else {
			fallback.append(uri.begin() + slash + 1, uri.begin() + nextSlash);
		}
		fallback.push_back('/');
		slash = nextSlash;
	}
	return ret;
}
