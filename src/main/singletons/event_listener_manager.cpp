#include "../../precompiled.hpp"
#include "event_listener_manager.hpp"
#include <map>
#include <vector>
#include <vector>
#include <boost/noncopyable.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "job_dispatcher.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
using namespace Poseidon;

class Poseidon::EventListener : boost::noncopyable,
	public boost::enable_shared_from_this<EventListener>
{
private:
	const unsigned m_id;
	const boost::weak_ptr<const void> m_dependency;
	const EventListenerCallback m_callback;

public:
	EventListener(unsigned id,
		const boost::weak_ptr<const void> &dependency, const EventListenerCallback &callback)
		: m_id(id), m_dependency(dependency), m_callback(callback)
	{
		LOG_INFO("Created event listener for event ", m_id);
	}
	~EventListener(){
		LOG_INFO("Destroyed event listener for event ", m_id);
	}

public:
	boost::shared_ptr<const EventListenerCallback>
		lock(boost::shared_ptr<const void> &lockedDep) const
	{
		if((m_dependency < boost::weak_ptr<void>()) || (boost::weak_ptr<void>() < m_dependency)){
			lockedDep = m_dependency.lock();
			if(!lockedDep){
				return NULLPTR;
			}
		}
		return boost::shared_ptr<const EventListenerCallback>(shared_from_this(), &m_callback);
	}
};

namespace {

boost::shared_mutex g_mutex;
std::map<unsigned, std::vector<boost::weak_ptr<const EventListener> > > g_listeners;

class EventJob : public JobBase {
private:
	const boost::shared_ptr<const void> m_lockedDep;
	const boost::shared_ptr<const EventListenerCallback> m_callback;

	boost::shared_ptr<EventBaseWithoutId> m_event;

public:
	EventJob(boost::shared_ptr<const void> lockedDep,
		boost::shared_ptr<const EventListenerCallback> callback,
		boost::shared_ptr<EventBaseWithoutId> event)
		: m_lockedDep(STD_MOVE(lockedDep))
		, m_callback(STD_MOVE(callback))
		, m_event(STD_MOVE(event))
	{
	}

protected:
	void perform(){
		(*m_callback)(STD_MOVE(m_event));
	}
};

}

namespace Poseidon {

void logInvalidDynamicEventType(unsigned id){
	LOG_ERROR("Invalid dynamic event type: event id = ", id);
}

}

void EventListenerManager::start(){
}
void EventListenerManager::stop(){
	LOG_INFO("Removing all event listeners...");

	g_listeners.clear();
}

void EventListenerManager::raise(const boost::shared_ptr<EventBaseWithoutId> &event){
	const unsigned eventId = event->id();

	std::vector<boost::shared_ptr<const EventListener> > listeners;
	bool needsCleanup = false;
	{
		const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
		const AUTO(it, g_listeners.find(eventId));
		if(it != g_listeners.end()){
			for(AUTO(it2, it->second.begin()); it2 != it->second.end(); ++it2){
				AUTO(listener, it2->lock());
				if(listener){
					listeners.push_back(STD_MOVE(listener));
				} else {
					needsCleanup = true;
				}
			}
		}
	}
	for(AUTO(it, listeners.begin()); it != listeners.end(); ++it){
		boost::shared_ptr<const void> lockedDep;
		AUTO(callback, (*it)->lock(lockedDep));
		if(!callback){
			continue;
		}
		boost::make_shared<EventJob>(
			STD_MOVE(lockedDep), STD_MOVE(callback), boost::ref(event))->pend();
	}
	if(needsCleanup){
		LOG_DEBUG("Cleaning up event listener list for event ", eventId);

		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		AUTO_REF(listenerList, g_listeners[eventId]);
		AUTO(it, listenerList.begin());
		while(it != listenerList.end()){
			if(it->expired()){
				it = listenerList.erase(it);
			} else {
				++it;
			}
		}
	}
}

boost::shared_ptr<const EventListener> EventListenerManager::doRegisterListener(unsigned id,
	const boost::weak_ptr<const void> &dependency, const EventListenerCallback &callback)
{
	AUTO(newServlet, boost::make_shared<EventListener>(
		id, boost::ref(dependency), boost::ref(callback)));
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		g_listeners[id].push_back(newServlet);
	}
	return newServlet;
}
