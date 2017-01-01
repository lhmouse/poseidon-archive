// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_EVENT_DISPATCHER_HPP_
#define POSEIDON_SINGLETONS_EVENT_DISPATCHER_HPP_

#include "../cxx_ver.hpp"
#include <typeinfo>
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

class EventDispatcher {
private:
	EventDispatcher();

public:
	static void start();
	static void stop();

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	static boost::shared_ptr<EventListener> register_listener(unsigned id, EventListenerCallback callback);

	// void (boost::shared_ptr<EventT> event)
	template<typename EventT>
	static
		typename boost::enable_if_c<boost::is_base_of<EventBaseWithoutId, EventT>::value,
			boost::shared_ptr<EventListener> >::type
		register_listener(boost::function<void (boost::shared_ptr<EventT>)> callback)
	{
		struct Helper {
			static void check_and_forward(boost::function<void (boost::shared_ptr<EventT>)> &callback,
				const boost::shared_ptr<EventBaseWithoutId> &event)
			{
				AUTO(derived, boost::dynamic_pointer_cast<EventT>(event));
				if(!derived){
					LOG_POSEIDON_ERROR("Incorrect dynamic event type: id = ", event->get_id(),
						", expecting = ", typeid(EventT).name(), ", typeid = ", typeid(*event.get()).name());
					DEBUG_THROW(Exception, sslit("Incorrect dynamic event type"));
				}
				callback(STD_MOVE(derived));
			}
		};
		return register_listener(EventT::ID, boost::bind(&Helper::check_and_forward, STD_MOVE_IDN(callback), _1));
	}

	static void sync_raise(const boost::shared_ptr<EventBaseWithoutId> &event);
	static void async_raise(const boost::shared_ptr<EventBaseWithoutId> &event, const boost::shared_ptr<const bool> &withdrawn);
};

}

#endif
