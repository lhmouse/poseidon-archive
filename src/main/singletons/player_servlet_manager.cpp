#include "../precompiled.hpp"
#include "player_servlet_manager.hpp"
#include <map>
#include <boost/noncopyable.hpp>
#include <boost/ref.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/locks.hpp>
#include "main_config.hpp"
#include "../log.hpp"
#include "../exception.hpp"
using namespace Poseidon;

namespace {

class RealPlayerServlet : boost::noncopyable,
	public boost::enable_shared_from_this<RealPlayerServlet>
{
private:
	const boost::uint16_t m_protocolId;
	const boost::weak_ptr<const void> m_dependency;
	const PlayerServletCallback m_callback;

public:
	RealPlayerServlet(boost::uint16_t protocolId,
		const boost::weak_ptr<const void> &dependency, const PlayerServletCallback &callback)
		: m_protocolId(protocolId), m_dependency(dependency), m_callback(callback)
	{
		LOG_INFO("Created player servlet for protocol ", m_protocolId);
	}
	~RealPlayerServlet(){
		LOG_INFO("Destroyed player servlet for protocol ", m_protocolId);
	}

public:
	boost::shared_ptr<const PlayerServletCallback> lock(boost::shared_ptr<const void> &lockedDep) const {
		if((m_dependency < boost::weak_ptr<void>()) || (boost::weak_ptr<void>() < m_dependency)){
			lockedDep = m_dependency.lock();
			if(!lockedDep){
				return VAL_INIT;
			}
		}
		return boost::shared_ptr<const PlayerServletCallback>(shared_from_this(), &m_callback);
	}
};

}

struct Poseidon::PlayerServlet : boost::noncopyable {
	const boost::shared_ptr<RealPlayerServlet> realPlayerServlet;

	explicit PlayerServlet(boost::shared_ptr<RealPlayerServlet> realPlayerServlet_)
		: realPlayerServlet(STD_MOVE(realPlayerServlet_))
	{
	}
};

namespace {

std::size_t g_maxRequestLength = 16 * 0x400;

boost::shared_mutex g_mutex;
std::map<unsigned, std::map<boost::uint16_t, boost::weak_ptr<const PlayerServlet> > > g_servlets;

}

void PlayerServletManager::start(){
	MainConfig::get(g_maxRequestLength, "player_max_request_length");
	LOG_DEBUG("Max request length = ", g_maxRequestLength);
}
void PlayerServletManager::stop(){
	LOG_INFO("Unloading all player servlets...");

	g_servlets.clear();
}

std::size_t PlayerServletManager::getMaxRequestLength(){
	return g_maxRequestLength;
}

boost::shared_ptr<PlayerServlet> PlayerServletManager::registerServlet(
	unsigned port, boost::uint16_t protocolId,
	const boost::weak_ptr<const void> &dependency, const PlayerServletCallback &callback)
{
	AUTO(newServlet, boost::make_shared<PlayerServlet>(boost::make_shared<RealPlayerServlet>(
		protocolId, boost::ref(dependency), boost::ref(callback))));
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		AUTO_REF(servlet, g_servlets[port][protocolId]);
		if(!servlet.expired()){
			LOG_ERROR("Duplicate player protocol servlet for id ", protocolId, " on port ", port);
			DEBUG_THROW(Exception, "Duplicate player protocol servlet");
		}
		servlet = newServlet;
	}
	return newServlet;
}

boost::shared_ptr<const PlayerServletCallback> PlayerServletManager::getServlet(
	unsigned port, boost::shared_ptr<const void> &lockedDep, boost::uint16_t protocolId)
{
	const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
	const AUTO(it, g_servlets.find(port));
	if(it == g_servlets.end()){
		LOG_DEBUG("No servlet on port ", port);
		return VAL_INIT;
	}

	AUTO_REF(servletsOnPort, it->second);
	const AUTO(sit, servletsOnPort.find(protocolId));
	if(sit == servletsOnPort.end()){
		LOG_DEBUG("No servlet for protocol ", protocolId);
		return VAL_INIT;
	}
	const AUTO(servlet, sit->second.lock());
	if(!servlet){
		LOG_DEBUG("Servlet expired");
		return VAL_INIT;
	}
	return servlet->realPlayerServlet->lock(lockedDep);
}
