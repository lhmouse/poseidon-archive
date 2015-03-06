// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "upgraded_session_base.hpp"
#include "session.hpp"

namespace Poseidon {

namespace Http {
	UpgradedSessionBase::UpgradedSessionBase(const boost::shared_ptr<Session> &parent)
		: m_parent(parent)
	{
	}

	void UpgradedSessionBase::onClose() NOEXCEPT {
	}

	bool UpgradedSessionBase::hasBeenShutdown() const {
		const AUTO(parent, getParent());
		if(!parent){
			return true;
		}
		return static_cast<const TcpSessionBase &>(*parent).hasBeenShutdown();
	}
	bool UpgradedSessionBase::send(StreamBuffer buffer, bool fin){
		const AUTO(parent, getParent());
		if(!parent){
			return false;
		}
		return static_cast<TcpSessionBase &>(*parent).send(STD_MOVE(buffer), fin);
	}
	bool UpgradedSessionBase::forceShutdown() NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return false;
		}
		return static_cast<TcpSessionBase &>(*parent).forceShutdown();
	}

	std::size_t UpgradedSessionBase::getCategory() const {
		return getSafeParent()->m_category;
	}
	const std::string &UpgradedSessionBase::getUri() const {
		return getSafeParent()->m_uri;
	}
	const OptionalMap &UpgradedSessionBase::getGetParams() const {
		return getSafeParent()->m_getParams;
	}
	const OptionalMap &UpgradedSessionBase::getHeaders() const {
		return getSafeParent()->m_headers;
	}

	void UpgradedSessionBase::setTimeout(unsigned long long timeout){
		getSafeParent()->setTimeout(timeout);
	}
	void UpgradedSessionBase::registerOnClose(boost::function<void ()> callback){
		getSafeParent()->registerOnClose(STD_MOVE(callback));
	}
}

}
