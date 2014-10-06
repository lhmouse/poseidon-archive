#include "../precompiled.hpp"
#include "event_base.hpp"
#include "singletons/event_listener_manager.hpp"
using namespace Poseidon;

EventBaseWithoutId::~EventBaseWithoutId(){
}

void EventBaseWithoutId::raise(){
	EventListenerManager::raise(shared_from_this());
}
