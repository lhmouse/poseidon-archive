#include "../../precompiled.hpp"
#include "upgraded_session_base.hpp"
#include "session.hpp"
#include "../optional_map.hpp"
using namespace Poseidon;

HttpUpgradedSessionBase::HttpUpgradedSessionBase(boost::weak_ptr<HttpSession> parent)
	: m_parent(STD_MOVE(parent))
{
}

void HttpUpgradedSessionBase::sendUsingMove(StreamBuffer &buffer){
	const AUTO(parent, m_parent.lock());
	if(!parent){
		return;
	}
	parent->sendUsingMove(buffer);
}
bool HttpUpgradedSessionBase::hasBeenShutdown() const {
	const AUTO(parent, m_parent.lock());
	if(!parent){
		return true;
	}
	return parent->hasBeenShutdown();
}
void HttpUpgradedSessionBase::shutdown(){
	const AUTO(parent, m_parent.lock());
	if(!parent){
		return;
	}
	parent->shutdown();
}
void HttpUpgradedSessionBase::forceShutdown(){
	const AUTO(parent, m_parent.lock());
	if(!parent){
		return;
	}
	parent->forceShutdown();
}

const std::string &HttpUpgradedSessionBase::getUri() const {
	return getParent()->m_uri;
}
const OptionalMap &HttpUpgradedSessionBase::getParams() const {
	return getParent()->m_getParams;
}
