// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "event_dispatcher.hpp"
#include "job_dispatcher.hpp"
#include "../event_base.hpp"
#include "../log.hpp"
#include "../mutex.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace {
	struct TypeInfoComparator {
		bool operator()(const std::type_info *lhs, const std::type_info *rhs) const NOEXCEPT {
			return (*lhs).before(*rhs);
		}
	};
	typedef boost::container::flat_multimap<const std::type_info *,
		boost::weak_ptr<const EventListenerCallback>, TypeInfoComparator> ListenerMap;

	Mutex g_mutex;
	ListenerMap g_listeners;

	class EventJob : public JobBase {
	private:
		const boost::shared_ptr<const EventListenerCallback> m_listener;
		const boost::shared_ptr<EventBase> m_event;

	public:
		EventJob(boost::shared_ptr<const EventListenerCallback> listener, boost::shared_ptr<EventBase> event)
			: m_listener(STD_MOVE(listener)), m_event(STD_MOVE(event))
		{ }

	protected:
		boost::weak_ptr<const void> get_category() const OVERRIDE {
			return m_event;
		}
		void perform() OVERRIDE {
			PROFILE_ME;

			(*m_listener)(m_event);
		}
	};

	void get_listeners(std::vector<boost::shared_ptr<const EventListenerCallback> > &listeners, const std::type_info *type_info){
		PROFILE_ME;

		const Mutex::UniqueLock lock(g_mutex);
		const AUTO(range, g_listeners.equal_range(type_info));
		listeners.reserve(listeners.size() + static_cast<std::size_t>(std::distance(range.first, range.second)));
		AUTO(it, range.first);
		while(it != range.second){
			AUTO(listener, it->second.lock());
			if(!listener){
				it = g_listeners.erase(it);
				continue;
			}
			listeners.push_back(STD_MOVE_IDN(listener));
			++it;
		}
	}
}

void EventDispatcher::start(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting event dispatcher...");
}
void EventDispatcher::stop(){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Removing all event listener callbacks...");

	g_listeners.clear();
}

boost::shared_ptr<const EventListenerCallback> EventDispatcher::register_listener_explicit(
	const std::type_info &type_info, EventListenerCallback callback)
{
	PROFILE_ME;

	AUTO(listener, boost::make_shared<EventListenerCallback>(STD_MOVE_IDN(callback)));
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_listeners.emplace(&type_info, listener);
	}
	return STD_MOVE_IDN(listener);
}

void EventDispatcher::sync_raise(const boost::shared_ptr<EventBase> &event){
	PROFILE_ME;

	std::vector<boost::shared_ptr<const EventListenerCallback> > listeners;
	get_listeners(listeners, &typeid(*event));
	for(AUTO(it, listeners.begin()); it != listeners.end(); ++it){
		(**it)(event);
	}
}
void EventDispatcher::async_raise(const boost::shared_ptr<EventBase> &event, const boost::shared_ptr<const bool> &withdrawn){
	PROFILE_ME;

	std::vector<boost::shared_ptr<const EventListenerCallback> > listeners;
	get_listeners(listeners, &typeid(*event));
	for(AUTO(it, listeners.begin()); it != listeners.end(); ++it){
		JobDispatcher::enqueue(boost::make_shared<EventJob>(STD_MOVE_IDN(*it), event), withdrawn);
	}
}

}
