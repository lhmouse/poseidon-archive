#include "../../precompiled.hpp"
#include "http_servlet_manager.hpp"
#include <string>
#include <map>
#include <boost/noncopyable.hpp>
#include <boost/ref.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/locks.hpp>
#include "../log.hpp"
#include "../exception.hpp"
using namespace Poseidon;

namespace {

const boost::weak_ptr<void> NULL_WEAK_PTR;

}

class Poseidon::HttpServlet : boost::noncopyable,
	public boost::enable_shared_from_this<HttpServlet>
{
private:
	const std::string m_uri;
	const boost::weak_ptr<void> m_dependency;
	const HttpServletCallback m_callback;

public:
	HttpServlet(std::string uri, boost::weak_ptr<void> dependency, HttpServletCallback callback)
		: m_uri(uri), m_dependency(dependency), m_callback(callback)
	{
		LOG_INFO("Created http servlet for URI ", m_uri);
	}
	~HttpServlet(){
		LOG_INFO("Destroyed http servlet for URI ", m_uri);
	}

public:
	boost::shared_ptr<const HttpServletCallback> lock() const {
		if(!(m_dependency < NULL_WEAK_PTR) && !(NULL_WEAK_PTR < m_dependency)){
			return boost::shared_ptr<const HttpServletCallback>(shared_from_this(), &m_callback);
		}
		const AUTO(lockedDep, m_dependency.lock());
		if(!lockedDep){
			return boost::shared_ptr<const HttpServletCallback>();
		}
		return boost::shared_ptr<const HttpServletCallback>(
			boost::make_shared<
				std::pair<boost::shared_ptr<void>, boost::shared_ptr<const HttpServlet> >
				>(lockedDep, shared_from_this()),
			&m_callback
		);
	}
};

namespace {

boost::shared_mutex g_mutex;
std::map<std::string, boost::weak_ptr<const HttpServlet> > g_servlets;

}

boost::shared_ptr<const HttpServlet> HttpServletManager::registerServlet(const std::string &uri,
	boost::weak_ptr<void> dependency, const HttpServletCallback &callback)
{
	const AUTO(newServlet, boost::make_shared<HttpServlet>(
		uri, boost::ref(dependency), boost::ref(callback)));
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		AUTO_REF(servlet, g_servlets[uri]);
		if(!servlet.expired()){
			DEBUG_THROW(Exception, "Duplicate http servlet.");
		}
		servlet = newServlet;
	}
	return newServlet;
}

boost::shared_ptr<const HttpServletCallback> HttpServletManager::getServlet(const std::string &uri){
	boost::shared_lock<boost::shared_mutex> slock(g_mutex);
	const AUTO(it, g_servlets.find(uri));
	if(it == g_servlets.end()){
		return boost::shared_ptr<const HttpServletCallback>();
	}
	const AUTO(servlet, it->second.lock());
	if(!servlet){
		return boost::shared_ptr<const HttpServletCallback>();
	}
	return servlet->lock();
}
