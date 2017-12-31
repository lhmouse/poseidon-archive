// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_EVENT_DISPATCHER_HPP_
#define POSEIDON_SINGLETONS_EVENT_DISPATCHER_HPP_

#include "../cxx_ver.hpp"
#include <typeinfo>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include "../exception.hpp"
#include "../log.hpp"

namespace Poseidon {

class EventListener; // 没有定义的类，当作句柄使用。
class EventBase;

class EventDispatcher {
private:
	EventDispatcher();

public:
	typedef boost::function<void (const boost::shared_ptr<EventBase> &event)> EventListenerCallback;

	static void start();
	static void stop();

	static void get_listeners(boost::container::vector<boost::shared_ptr<const EventListener> > &ret, const std::type_info &type_inf);

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	static boost::shared_ptr<const EventListener> register_listener_explicit(const std::type_info &type_info, EventListenerCallback callback);

	template<typename EventT>
	static boost::shared_ptr<const EventListener> register_listener(boost::function<void (const boost::shared_ptr<EventT> &)> callback){
		struct Helper {
			static void safe_fwd(boost::function<void (const boost::shared_ptr<EventT> &)> &callback, const boost::shared_ptr<EventBase> &event){
				AUTO(derived, boost::dynamic_pointer_cast<EventT>(event));
				if(!derived){
					LOG_POSEIDON_ERROR("Incorrect dynamic event type: expecting ", typeid(EventT).name(), ", got ", typeid(*event).name());
					DEBUG_THROW(Exception, sslit("Incorrect dynamic event type"));
				}
				callback(STD_MOVE(derived));
			}
		};
		return register_listener_explicit(typeid(EventT), boost::bind(&Helper::safe_fwd, STD_MOVE_IDN(callback), _1));
	}

	static void sync_raise(const boost::shared_ptr<EventBase> &event);
	static void async_raise(const boost::shared_ptr<EventBase> &event, const boost::shared_ptr<const bool> &withdrawn);
};

}

#endif
