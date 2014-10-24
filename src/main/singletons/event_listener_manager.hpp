#ifndef POSEIDON_SINGLETONS_EVENT_LISTENER_MANAGER_HPP_
#define POSEIDON_SINGLETONS_EVENT_LISTENER_MANAGER_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include "../event_base.hpp"

#ifdef POSEIDON_CXX11
#   include <functional>
#else
#   include <tr1/functional>
#endif

namespace Poseidon {

class EventListener;

typedef TR1::function<
	void (boost::shared_ptr<EventBaseWithoutId> event)
	> EventListenerCallback;

struct EventListenerManager {
	static void start();
	static void stop();

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	// callback 禁止 move，否则可能出现主模块中引用子模块内存的情况。
	template<class EventT>
	static boost::shared_ptr<EventListener> registerListener(
		const boost::weak_ptr<const void> &dependency,
		const TR1::function<void (boost::shared_ptr<EventT>)> &callback)
	{
		BOOST_STATIC_ASSERT((boost::is_base_of<EventBaseWithoutId, EventT>::value));

		struct Helper {
			static void checkedForward(boost::shared_ptr<EventBaseWithoutId> event,
				const TR1::function<void (boost::shared_ptr<EventT>)> &derivedCallback)
			{
				AUTO(derived, boost::dynamic_pointer_cast<EventT>(event));
				if(!derived){
					logInvalidDynamicEventType(EventT::EVENT_ID);
					return;
				}
				derivedCallback(STD_MOVE(derived));
			}
		};
		return doRegisterListener(EventT::EVENT_ID, dependency,
			TR1::bind(&Helper::checkedForward, TR1::placeholders::_1, callback));
	}

	static void raise(const boost::shared_ptr<EventBaseWithoutId> &event);

private:
	EventListenerManager();

	static boost::shared_ptr<EventListener> doRegisterListener(unsigned id,
		const boost::weak_ptr<const void> &dependency, const EventListenerCallback &callback);

	static void logInvalidDynamicEventType(unsigned id);
};

}

#endif
