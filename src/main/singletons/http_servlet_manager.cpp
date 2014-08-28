#include "../../precompiled.hpp"
#include "http_servlet_manager.hpp"
#include <string>
#include <map>
#include <boost/noncopyable.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/locks.hpp>
#include "../log.hpp"
#include "../exception.hpp"
using namespace Poseidon;

namespace {

typedef HttpServletManager::HttpServletCallback HttpServletCallback;

}

struct Poseidon::HttpServlet : boost::noncopyable {
	const std::string m_uri;
	const HttpServletCallback m_callback;

	HttpServlet(std::string uri, HttpServletCallback callback)
		: m_uri(uri), m_callback(callback)
	{
		LOG_INFO("Created http servlet for URI ", m_uri);
	}
	~HttpServlet(){
		LOG_INFO("Destroyed http servlet for URI ", m_uri);
	}
};

namespace {

boost::shared_mutex g_mutex;
std::map<std::string,
	std::pair<boost::weak_ptr<void>, boost::weak_ptr<HttpServlet> >
	> g_servlets;

const boost::weak_ptr<void> NULL_WEAK_PTR;

}

boost::shared_ptr<HttpServlet> HttpServletManager::registerServlet(const std::string &uri,
	boost::weak_ptr<void> dependency, HttpServletCallback callback)
{
	const AUTO(newServlet, boost::make_shared<HttpServlet>(uri, callback));
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		AUTO_REF(pair, g_servlets[uri]);
		if(!pair.second.expired()){
			DEBUG_THROW(Exception, "Duplicate http servlet.");
		}
		pair.first = dependency;
		pair.second = newServlet;
	}
	return newServlet;
}

boost::shared_ptr<const HttpServletCallback> HttpServletManager::getServlet(const std::string &uri){
	boost::shared_lock<boost::shared_mutex> slock(g_mutex);
	const AUTO(it, g_servlets.find(uri));
	if(it == g_servlets.end()){
		return boost::shared_ptr<HttpServletCallback>();
	}
	if(!(it->second.first < NULL_WEAK_PTR) && !(NULL_WEAK_PTR < it->second.first)){
		const AUTO(servlet, it->second.second.lock());
		if(!servlet){
			slock.unlock();
			const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
			g_servlets.erase(it);
			return boost::shared_ptr<HttpServletCallback>();
		}
		return boost::shared_ptr<const HttpServletCallback>(servlet, &(servlet->m_callback));
	}
	const AUTO(dependency, it->second.first.lock());
	const AUTO(servlet, it->second.second.lock());
	if(!dependency || !servlet){
		slock.unlock();
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		g_servlets.erase(it);
		return boost::shared_ptr<HttpServletCallback>();
	}
	const AUTO(locked, (boost::make_shared<
		std::pair<boost::shared_ptr<void>, boost::shared_ptr<HttpServlet> >
	>(dependency, servlet)));
	return boost::shared_ptr<const HttpServletCallback>(locked, &(locked->second->m_callback));
}
