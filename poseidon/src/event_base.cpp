// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "event_base.hpp"
#include "singletons/event_dispatcher.hpp"

namespace Poseidon {

Event_base::~Event_base(){
	//
}

void sync_raise_event(const boost::shared_ptr<Event_base> &event){
	Event_dispatcher::sync_raise(event);
}
void async_raise_event(const boost::shared_ptr<Event_base> &event, const boost::shared_ptr<const bool> &withdrawn){
	Event_dispatcher::async_raise(event, withdrawn);
}

}
