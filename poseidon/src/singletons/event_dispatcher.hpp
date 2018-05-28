// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_EVENT_DISPATCHER_HPP_
#define POSEIDON_SINGLETONS_EVENT_DISPATCHER_HPP_

#include "../cxx_ver.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include <typeinfo>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/container/vector.hpp>

namespace Poseidon {

class Event_listener; // 没有定义的类，当作句柄使用。
class Event_base;

class Event_dispatcher {
private:
	Event_dispatcher();

public:
	typedef std::function<void (const boost::shared_ptr<Event_base> &event)> Event_listener_callback;

	static void start();
	static void stop();

	static void get_listeners(boost::container::vector<boost::shared_ptr<const Event_listener> > &ret, const std::type_info &type_inf);

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	static boost::shared_ptr<const Event_listener> register_listener_explicit(const std::type_info &type_info, Event_listener_callback callback);

	template<typename EventT>
	static boost::shared_ptr<const Event_listener> register_listener(std::function<void (const boost::shared_ptr<EventT> &)> callback){
		struct Helper {
			static void safe_fwd(std::function<void (const boost::shared_ptr<EventT> &)> &callback, const boost::shared_ptr<Event_base> &event){
				AUTO(derived, boost::dynamic_pointer_cast<EventT>(event));
				if(!derived){
					POSEIDON_LOG_ERROR("Incorrect dynamic event type: expecting ", typeid(EventT).name(), ", got ", typeid(*event).name());
					POSEIDON_THROW(Exception, Rcnts::view("Incorrect dynamic event type"));
				}
				callback(STD_MOVE(derived));
			}
		};
		return register_listener_explicit(typeid(EventT), std::bind(&Helper::safe_fwd, STD_MOVE_IDN(callback), _1));
	}

	static void sync_raise(const boost::shared_ptr<Event_base> &event);
	static void async_raise(const boost::shared_ptr<Event_base> &event, const boost::shared_ptr<const bool> &withdrawn);
};

}

#endif
