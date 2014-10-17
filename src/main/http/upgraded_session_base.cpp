#include "../../precompiled.hpp"
#include "upgraded_session_base.hpp"
#include "session.hpp"
#include "../optional_map.hpp"
using namespace Poseidon;

HttpUpgradedSessionBase::HttpUpgradedSessionBase(boost::weak_ptr<HttpSession> parent)
	: m_parent(STD_MOVE(parent))
{
}

void HttpUpgradedSessionBase::onInitContents(const void *data, std::size_t size){
	(void)data;
	(void)size;
}

const std::string &HttpUpgradedSessionBase::getRemoteIp() const {
	return getSafeParent()->getRemoteIp();
}
bool HttpUpgradedSessionBase::send(StreamBuffer buffer){
	const AUTO(parent, getParent());
	if(!parent){
		return false;
	}
	return parent->TcpSessionBase::send(STD_MOVE(buffer));
}
bool HttpUpgradedSessionBase::hasBeenShutdown() const {
	const AUTO(parent, getParent());
	if(!parent){
		return true;
	}
	return parent->TcpSessionBase::hasBeenShutdown();
}
bool HttpUpgradedSessionBase::shutdown(){
	const AUTO(parent, getParent());
	if(!parent){
		return false;
	}
	return parent->TcpSessionBase::shutdown();
}
bool HttpUpgradedSessionBase::shutdown(StreamBuffer buffer){
	const AUTO(parent, getParent());
	if(!parent){
		return false;
	}
	return parent->TcpSessionBase::shutdown(STD_MOVE(buffer));
}
bool HttpUpgradedSessionBase::forceShutdown(){
	const AUTO(parent, getParent());
	if(!parent){
		return false;
	}
	return parent->TcpSessionBase::forceShutdown();
}

const std::string &HttpUpgradedSessionBase::getUri() const {
	return getSafeParent()->m_uri;
}
const OptionalMap &HttpUpgradedSessionBase::getParams() const {
	return getSafeParent()->m_getParams;
}
const OptionalMap &HttpUpgradedSessionBase::getHeaders() const {
	return getSafeParent()->m_headers;
}
