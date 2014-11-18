// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "event_base.hpp"
#include "singletons/event_listener_manager.hpp"
using namespace Poseidon;

EventBaseWithoutId::~EventBaseWithoutId(){
}

void EventBaseWithoutId::raise(){
	EventListenerManager::raise(shared_from_this());
}
