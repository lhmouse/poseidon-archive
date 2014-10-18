#include "../../precompiled.hpp"
#include "upgraded_session_base.hpp"
#include "session.hpp"
#include "../optional_map.hpp"
using namespace Poseidon;

HttpUpgradedSessionBase::HttpUpgradedSessionBase(const boost::shared_ptr<HttpSession> &parent)
	: m_parent(parent), m_remoteIp(parent->getRemoteIp())
	, m_uri(parent->m_uri), m_getParams(parent->m_getParams), m_headers(parent->m_headers)
{
}

void HttpUpgradedSessionBase::onInitContents(const void *data, std::size_t size){
	(void)data;
	(void)size;
}

const std::string &HttpUpgradedSessionBase::getRemoteIp() const {
	return m_remoteIp;
}
bool HttpUpgradedSessionBase::hasBeenShutdown() const {
	const AUTO(parent, getParent());
	if(!parent){
		return false;
	}
	return static_cast<const TcpSessionBase *>(parent.get())->hasBeenShutdown();
}
bool HttpUpgradedSessionBase::send(StreamBuffer buffer, bool final){
	const AUTO(parent, getParent());
	if(!parent){
		return false;
	}
	return static_cast<TcpSessionBase *>(parent.get())->send(STD_MOVE(buffer), final);
}
bool HttpUpgradedSessionBase::forceShutdown(){
	const AUTO(parent, getParent());
	if(!parent){
		return false;
	}
	return static_cast<TcpSessionBase *>(parent.get())->forceShutdown();
}
