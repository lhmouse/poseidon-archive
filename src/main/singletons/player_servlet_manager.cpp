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

struct Poseidon::PlayerServlet : boost::noncopyable {
	const boost::uint16_t protocolId;
	const boost::shared_ptr<const PlayerServletCallback> callback;

	PlayerServlet(boost::uint16_t protocolId_, boost::shared_ptr<const PlayerServletCallback> callback_)
		: protocolId(protocolId_), callback(STD_MOVE(callback_))
	{
	}
};

namespace {

std::size_t g_maxRequestLength = 16 * 0x400;

typedef std::map<unsigned,
	std::map<boost::uint16_t, boost::weak_ptr<const PlayerServlet> >
	> ServletMap;

boost::shared_mutex g_mutex;
ServletMap g_servlets;

}

void PlayerServletManager::start(){
	MainConfig::get(g_maxRequestLength, "player_max_request_length");
	LOG_DEBUG("Max request length = ", g_maxRequestLength);
}
void PlayerServletManager::stop(){
	LOG_INFO("Unloading all player servlets...");

	ServletMap servlets;
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		servlets.swap(servlets);
	}
}

std::size_t PlayerServletManager::getMaxRequestLength(){
	return g_maxRequestLength;
}

boost::shared_ptr<PlayerServlet> PlayerServletManager::registerServlet(
	unsigned port, boost::uint16_t protocolId, PlayerServletCallback callback)
{
	AUTO(sharedCallback, boost::make_shared<PlayerServletCallback>());
	sharedCallback->swap(callback);
	AUTO(servlet, boost::make_shared<PlayerServlet>(protocolId, sharedCallback));
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		AUTO_REF(old, g_servlets[port][protocolId]);
		if(!old.expired()){
			LOG_ERROR("Duplicate player protocol servlet for id ", protocolId, " on port ", port);
			DEBUG_THROW(Exception, "Duplicate player protocol servlet");
		}
		old = servlet;
	}
	return servlet;
}

boost::shared_ptr<const PlayerServletCallback> PlayerServletManager::getServlet(
	unsigned port, boost::uint16_t protocolId)
{
    const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
    const AUTO(it, g_servlets.find(port));
    if(it == g_servlets.end()){
        LOG_DEBUG("No servlet on port ", port);
        return VAL_INIT;
    }
    const AUTO(it2, it->second.find(protocolId));
    if(it2 == it->second.end()){
    	LOG_DEBUG("No servlet for protocol ", protocolId, " on port ", port);
    	return VAL_INIT;
    }
    const AUTO(servlet, it2->second.lock());
    if(!servlet){
    	LOG_DEBUG("Servlet for protocol ", protocolId, " on port ", port, " has expired");
    	return VAL_INIT;
    }
    return servlet->callback;
}
