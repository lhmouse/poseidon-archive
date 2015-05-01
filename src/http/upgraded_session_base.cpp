// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "upgraded_session_base.hpp"
#include "session.hpp"

namespace Poseidon {

namespace Http {
	UpgradedSessionBase::UpgradedSessionBase(const boost::shared_ptr<Session> &parent, std::string uri)
		: m_parent(parent), m_uri(STD_MOVE(uri))
	{
	}

	void UpgradedSessionBase::onInit(RequestHeaders requestHeaders, StreamBuffer entity){
		(void)requestHeaders;
		(void)entity;
	}

	void UpgradedSessionBase::onClose() NOEXCEPT {
	}
	void UpgradedSessionBase::onReadHup() NOEXCEPT {
	}

	bool UpgradedSessionBase::send(StreamBuffer buffer, bool fin){
		const AUTO(parent, getParent());
		if(!parent){
			return false;
		}
		return static_cast<TcpSessionBase &>(*parent).send(STD_MOVE(buffer), fin);
	}

	bool UpgradedSessionBase::hasBeenShutdown() const {
		const AUTO(parent, getParent());
		if(!parent){
			return true;
		}
		return static_cast<const TcpSessionBase &>(*parent).hasBeenShutdown();
	}
	bool UpgradedSessionBase::shutdown() NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return false;
		}
		return static_cast<TcpSessionBase &>(*parent).shutdown();
	}
	bool UpgradedSessionBase::forceShutdown() NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return false;
		}
		return static_cast<TcpSessionBase &>(*parent).forceShutdown();
	}

	void UpgradedSessionBase::setTimeout(boost::uint64_t timeout){
		getSafeParent()->setTimeout(timeout);
	}
	void UpgradedSessionBase::registerOnClose(boost::function<void ()> callback){
		getSafeParent()->registerOnClose(STD_MOVE(callback));
	}
}

}
