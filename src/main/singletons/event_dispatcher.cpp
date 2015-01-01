// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "event_dispatcher.hpp"
#include <map>
#include <vector>
#include <vector>
#include <boost/thread/shared_mutex.hpp>
#include "job_dispatcher.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
using namespace Poseidon;

struct Poseidon::EventListener : NONCOPYABLE {
	const unsigned id;
	const boost::shared_ptr<const EventListenerCallback> callback;

	EventListener(unsigned id_, boost::shared_ptr<const EventListenerCallback> callback_)
		: id(id_), callback(STD_MOVE(callback_))
	{
		LOG_POSEIDON_INFO("Created event listener for event ", id);
	}
	~EventListener(){
		LOG_POSEIDON_INFO("Destroyed event listener for event ", id);
	}
};

namespace {

typedef std::map<unsigned,
	std::vector<boost::weak_ptr<EventListener> >
	> ListenerMap;

boost::shared_mutex g_mutex;
ListenerMap g_listeners;

class EventJob : public JobBase {
private:
	const boost::shared_ptr<const EventListenerCallback> m_callback;
	boost::shared_ptr<EventBaseWithoutId> m_event;

public:
	EventJob(boost::shared_ptr<const EventListenerCallback> callback,
		boost::shared_ptr<EventBaseWithoutId> event)
		: m_callback(STD_MOVE(callback)), m_event(STD_MOVE(event))
	{
	}

protected:
	void perform(){
		PROFILE_ME;

		(*m_callback)(STD_MOVE(m_event));
	}
};

}

void EventDispatcher::start(){
}
void EventDispatcher::stop(){
	LOG_POSEIDON_INFO("Removing all event listeners...");

	ListenerMap listeners;
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		listeners.swap(g_listeners);
	}
}

boost::shared_ptr<EventListener> EventDispatcher::registerListener(
	unsigned id, EventListenerCallback callback)
{
	AUTO(sharedCallback, boost::make_shared<EventListenerCallback>());
	sharedCallback->swap(callback);
	AUTO(listener, boost::make_shared<EventListener>(id, sharedCallback));
	{
		const boost::unique_lock<boost::shared_mutex> ulock(g_mutex);
		g_listeners[id].push_back(listener);
	}
	return listener;
}

void EventDispatcher::raise(const boost::shared_ptr<EventBaseWithoutId> &event){
	const unsigned eventId = event->id();
	std::vector<boost::shared_ptr<const EventListenerCallback> > callbacks;
	bool needsCleanup = false;
	{
		const boost::shared_lock<boost::shared_mutex> slock(g_mutex);
		const AUTO(it, g_listeners.find(eventId));
		if(it != g_listeners.end()){
			for(AUTO(it2, it->second.begin()); it2 != it->second.end(); ++it2){
				const AUTO(listener, it2->lock());
				if(listener){
					LOG_POSEIDON_TRACE("Preparing an event job for dispatching: id = ", eventId);
					callbacks.push_back(listener->callback);
				} else {
					needsCleanup = true;
				}
			}
		}
	}
	for(AUTO(it, callbacks.begin()); it != callbacks.end(); ++it){
		pendJob(boost::make_shared<EventJob>(*it, boost::ref(event)));
	}
	if(needsCleanup){
		LOG_POSEIDON_DEBUG("Cleaning up event listener list for event ", eventId);

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
