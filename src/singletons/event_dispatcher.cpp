// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "event_dispatcher.hpp"
#include "job_dispatcher.hpp"
#include "../event_base.hpp"
#include "../log.hpp"
#include "../mutex.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

typedef Event_dispatcher::Event_listener_callback Event_listener_callback;

class Event_listener : NONCOPYABLE {
private:
	Event_listener_callback m_callback;

public:
	explicit Event_listener(Event_listener_callback callback)
		: m_callback(STD_MOVE_IDN(callback))
	{
		//
	}

public:
	const Event_listener_callback &get_callback() const {
		return m_callback;
	}
};

namespace {
	struct Type_info_comparator {
		bool operator()(const std::type_info *lhs, const std::type_info *rhs) const NOEXCEPT {
			return (*lhs).before(*rhs);
		}
	};
	typedef boost::container::flat_multimap<const std::type_info *, boost::weak_ptr<const Event_listener>, Type_info_comparator> Listener_map;

	Mutex g_mutex;
	Listener_map g_listeners;

	class Event_job : public Job_base {
	private:
		const boost::weak_ptr<const Event_listener> m_weak_listener;
		const boost::shared_ptr<Event_base> m_event;

	public:
		Event_job(boost::weak_ptr<const Event_listener> weak_listener, boost::shared_ptr<Event_base> event)
			: m_weak_listener(STD_MOVE(weak_listener)), m_event(STD_MOVE(event))
		{
			//
		}

	protected:
		boost::weak_ptr<const void> get_category() const OVERRIDE {
			return m_event;
		}
		void perform() OVERRIDE {
			PROFILE_ME;

			const AUTO(listener, m_weak_listener.lock());
			if(!listener){
				return;
			}
			listener->get_callback()(m_event);
		}
	};
}

void Event_dispatcher::start(){
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Starting event dispatcher...");

	//
}
void Event_dispatcher::stop(){
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Stopping event dispatcher...");

	const Mutex::Unique_lock lock(g_mutex);
	g_listeners.clear();
}

void Event_dispatcher::get_listeners(boost::container::vector<boost::shared_ptr<const Event_listener> > &ret, const std::type_info &type_info){
	PROFILE_ME;

	const Mutex::Unique_lock lock(g_mutex);
	const AUTO(range, g_listeners.equal_range(&type_info));
	ret.reserve(ret.size() + static_cast<std::size_t>(std::distance(range.first, range.second)));
	bool expired;
	for(AUTO(it, range.first); it != range.second; expired ? (it = g_listeners.erase(it)) : ++it){
		AUTO(listener, it->second.lock());
		expired = !listener;
		if(listener){
			ret.push_back(STD_MOVE_IDN(listener));
		}
	}
}

boost::shared_ptr<const Event_listener> Event_dispatcher::register_listener_explicit(const std::type_info &type_info, Event_listener_callback callback){
	PROFILE_ME;

	AUTO(listener, boost::make_shared<Event_listener>(STD_MOVE_IDN(callback)));
	{
		const Mutex::Unique_lock lock(g_mutex);
		g_listeners.emplace(&type_info, listener);
	}
	return STD_MOVE_IDN(listener);
}

void Event_dispatcher::sync_raise(const boost::shared_ptr<Event_base> &event){
	PROFILE_ME;

	boost::container::vector<boost::shared_ptr<const Event_listener> > listeners;
	get_listeners(listeners, typeid(*event));
	for(AUTO(it, listeners.begin()); it != listeners.end(); ++it){
		AUTO_REF(listener, *it);
		listener->get_callback()(event);
	}
}
void Event_dispatcher::async_raise(const boost::shared_ptr<Event_base> &event, const boost::shared_ptr<const bool> &withdrawn){
	PROFILE_ME;

	boost::container::vector<boost::shared_ptr<const Event_listener> > listeners;
	get_listeners(listeners, typeid(*event));
	for(AUTO(it, listeners.begin()); it != listeners.end(); ++it){
		AUTO_REF(listener, *it);
		Job_dispatcher::enqueue(boost::make_shared<Event_job>(STD_MOVE_IDN(listener), event), withdrawn);
	}
}

}
