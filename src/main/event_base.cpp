// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "event_base.hpp"
#include "singletons/event_dispatcher.hpp"

namespace Poseidon {

EventBaseWithoutId::~EventBaseWithoutId(){
}

void raiseEvent(const boost::shared_ptr<EventBaseWithoutId> &event){
	EventDispatcher::raise(event);
}

}
