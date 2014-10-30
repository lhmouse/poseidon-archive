#include "../precompiled.hpp"
#include "upgraded_session_base.hpp"
#include "session.hpp"
using namespace Poseidon;

HttpUpgradedSessionBase::HttpUpgradedSessionBase(const boost::shared_ptr<HttpSession> &parent)
	: m_parent(parent)
{
}

void HttpUpgradedSessionBase::onInitContents(const void *data, std::size_t size){
	(void)data;
	(void)size;
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

std::size_t HttpUpgradedSessionBase::getCategory() const {
	return getSafeParent()->m_category;
}
const std::string &HttpUpgradedSessionBase::getUri() const {
	return getSafeParent()->m_uri;
}
const OptionalMap &HttpUpgradedSessionBase::getGetParams() const {
	return getSafeParent()->m_getParams;
}
const OptionalMap &HttpUpgradedSessionBase::getHeaders() const {
	return getSafeParent()->m_headers;
}
