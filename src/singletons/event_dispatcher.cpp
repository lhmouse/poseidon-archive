// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "event_dispatcher.hpp"
#include "job_dispatcher.hpp"
#include "../log.hpp"
#include "../mutex.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

struct EventListener : NONCOPYABLE {
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

	Mutex g_mutex;
	ListenerMap g_listeners;

	class EventJob : public JobBase {
	private:
		const boost::shared_ptr<const EventListenerCallback> m_callback;
		const boost::shared_ptr<EventBaseWithoutId> m_event;

	public:
		EventJob(boost::shared_ptr<const EventListenerCallback> callback,
			boost::shared_ptr<EventBaseWithoutId> event)
			: m_callback(STD_MOVE(callback)), m_event(STD_MOVE(event))
		{
		}

	protected:
		boost::weak_ptr<const void> get_category() const OVERRIDE {
			return m_event;
		}
		void perform() OVERRIDE {
			PROFILE_ME;

			(*m_callback)(m_event);
		}
	};

	std::vector<boost::shared_ptr<const EventListenerCallback> > get_callbacks(const boost::shared_ptr<EventBaseWithoutId> &event){
		PROFILE_ME;

		std::vector<boost::shared_ptr<const EventListenerCallback> > callbacks;

		const Mutex::UniqueLock lock(g_mutex);
		const AUTO(it, g_listeners.find(event->id()));
		if(it != g_listeners.end()){
			AUTO(listener_it, it->second.begin());
			while(listener_it != it->second.end()){
				const AUTO(listener, listener_it->lock());
				if(!listener){
					listener_it = it->second.erase(listener_it);
					continue;
				}
				callbacks.push_back(listener->callback);
				++listener_it;
			}
		}
		return callbacks;
	}
}

void EventDispatcher::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting event dispatcher...");
}
void EventDispatcher::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Removing all event listeners...");

	ListenerMap listeners;
	{
		const Mutex::UniqueLock lock(g_mutex);
		listeners.swap(g_listeners);
	}
}

boost::shared_ptr<EventListener> EventDispatcher::register_listener(
	unsigned id, EventListenerCallback callback)
{
	AUTO(shared_callback, boost::make_shared<EventListenerCallback>());
	shared_callback->swap(callback);
	AUTO(listener, boost::make_shared<EventListener>(id, shared_callback));
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_listeners[id].push_back(listener);
	}
	return listener;
}

void EventDispatcher::sync_raise(const boost::shared_ptr<EventBaseWithoutId> &event){
	PROFILE_ME;

	const AUTO(callbacks, get_callbacks(event));
	for(AUTO(it, callbacks.begin()); it != callbacks.end(); ++it){
		(**it)(event);
	}
}
void EventDispatcher::async_raise(const boost::shared_ptr<EventBaseWithoutId> &event,
	const boost::shared_ptr<const bool> &withdrawn)
{
	PROFILE_ME;

	const AUTO(callbacks, get_callbacks(event));
	for(AUTO(it, callbacks.begin()); it != callbacks.end(); ++it){
		JobDispatcher::enqueue(boost::make_shared<EventJob>(*it, event), withdrawn);
	}
}

}
