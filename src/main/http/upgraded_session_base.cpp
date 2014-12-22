// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "upgraded_session_base.hpp"
#include "session.hpp"
using namespace Poseidon;

HttpUpgradedSessionBase::HttpUpgradedSessionBase(const boost::shared_ptr<HttpSession> &parent)
	: m_parent(parent)
{
}

bool HttpUpgradedSessionBase::hasBeenShutdown() const {
	const AUTO(parent, getParent());
	if(!parent){
		return false;
	}
	return static_cast<const TcpSessionBase *>(parent.get())->hasBeenShutdown();
}
bool HttpUpgradedSessionBase::send(StreamBuffer buffer, bool fin){
	const AUTO(parent, getParent());
	if(!parent){
		return false;
	}
	return static_cast<TcpSessionBase *>(parent.get())->send(STD_MOVE(buffer), fin);
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

void HttpUpgradedSessionBase::setTimeout(unsigned long long timeout){
	getSafeParent()->setTimeout(timeout);
}
