#ifndef POSEIDON_SINGLETONS_EVENT_LISTENER_MANAGER_HPP_
#define POSEIDON_SINGLETONS_EVENT_LISTENER_MANAGER_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>
#include "../event_base.hpp"
#include "../exception.hpp"
#include "../log.hpp"

namespace Poseidon {

class EventListener;

typedef boost::function<
	void (boost::shared_ptr<EventBaseWithoutId> event)
	> EventListenerCallback;

struct EventListenerManager {
	static void start();
	static void stop();

	static boost::shared_ptr<EventListener> registerListener(
		unsigned id, EventListenerCallback callback);

	template<typename EventT, typename CallbackT>
	static
		typename boost::enable_if_c<boost::is_base_of<EventBaseWithoutId, EventT>::value,
			boost::shared_ptr<EventListener> >::type
		registerListener(
#ifdef POSEIDON_CXX11
			CallbackT &&
#else
			const CallbackT &
#endif
			callback)
	{
		struct Helper {
			static void checkForward(
				boost::shared_ptr<EventBaseWithoutId> event, const CallbackT &callback)
			{
				AUTO(derived, boost::dynamic_pointer_cast<EventT>(event));
				if(!derived){
					LOG_ERROR("Invalid dynamic event: id = ", event->id());
					DEBUG_THROW(Exception, "Invalid dynamic event");
				}
				callback(STD_MOVE(derived));
			}
		};
		return registerListener(EventT::EVENT_ID, boost::bind(&Helper::checkForward, _1,
#ifdef POSEIDON_CXX11
			std::forward<CallbackT>(callback)
#else
			callback
#endif
			));
	}

	static void raise(const boost::shared_ptr<EventBaseWithoutId> &event);

private:
	EventListenerManager();
};

}

#endif
