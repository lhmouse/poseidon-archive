// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_EVENT_DISPATCHER_HPP_
#define POSEIDON_SINGLETONS_EVENT_DISPATCHER_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include "../event_base.hpp"
#include "../exception.hpp"
#include "../log.hpp"

namespace Poseidon {

class EventListener;

typedef boost::function<
	void (boost::shared_ptr<EventBaseWithoutId> event)
	> EventListenerCallback;

struct EventDispatcher {
	static void start();
	static void stop();

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	static boost::shared_ptr<EventListener> registerListener(unsigned id, EventListenerCallback callback);

	static void raise(const boost::shared_ptr<EventBaseWithoutId> &event,
		const boost::shared_ptr<const bool> &withdrawn = boost::shared_ptr<const bool>());

private:
	EventDispatcher();
};

}

#endif
