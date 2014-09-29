#include "../../precompiled.hpp"
#include "player_servlet_manager.hpp"
#include <map>
#include <boost/noncopyable.hpp>
#include <boost/ref.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/locks.hpp>
#include "../log.hpp"
#include "../exception.hpp"
using namespace Poseidon;

class Poseidon::PlayerServlet : boost::noncopyable,
	public boost::enable_shared_from_this<PlayerServlet>
{
private:
	const boost::uint16_t m_protocolId;
	const boost::weak_ptr<const void> m_dependency;
	const PlayerServletCallback m_callback;

public:
	PlayerServlet(boost::uint16_t protocolId,
		const boost::weak_ptr<const void> &dependency, const PlayerServletCallback &callback)
		: m_protocolId(protocolId), m_dependency(dependency), m_callback(callback)
	{
		LOG_INFO("Created player servlet for protocol ", m_protocolId);
	}
	~PlayerServlet(){
		LOG_INFO("Destroyed player servlet for protocol", m_protocolId);
	}

public:
	boost::shared_ptr<const PlayerServletCallback>
		lock(boost::shared_ptr<const void> &lockedDep) const
	{
		if((m_dependency < boost::weak_ptr<void>()) ||
			(boost::weak_ptr<void>() < m_dependency))
		{
			lockedDep = m_dependency.lock();
			if(!lockedDep){
				return NULLPTR;
			}
		}
		return boost::shared_ptr<const PlayerServletCallback>(
			shared_from_this(), &m_callback);
	}
};

namespace {

boost::shared_mutex g_mutex;
std::map<boost::uint16_t, boost::weak_ptr<const PlayerServlet> > g_servlets;

}

boost::shared_ptr<const PlayerServlet>
	PlayerServletManager::registerServlet(boost::uint16_t protocolId,
		const boost::weak_ptr<const void> &dependency, const PlayerServletCallback &callback)
{
	AUTO(newServlet, boost::make_shared<PlayerServlet>(
		protocolId, boost::ref(dependency), boost::ref(callback)));
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		AUTO_REF(servlet, g_servlets[protocolId]);
		if(!servlet.expired()){
			DEBUG_THROW(Exception, "Duplicate protocol servlet: " +
				boost::lexical_cast<std::string>(protocolId));
		}
		servlet = newServlet;
	}
	return newServlet;
}

boost::shared_ptr<const PlayerServletCallback>
	PlayerServletManager::getServlet(boost::shared_ptr<const void> &lockedDep, boost::uint16_t protocolId)
{
	const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
	const AUTO(it, g_servlets.find(protocolId));
	if(it == g_servlets.end()){
		return NULLPTR;
	}
	const AUTO(servlet, it->second.lock());
	if(!servlet){
		return NULLPTR;
	}
	return servlet->lock(lockedDep);
}
