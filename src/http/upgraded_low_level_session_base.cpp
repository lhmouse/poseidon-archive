// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "upgraded_low_level_session_base.hpp"
#include "session.hpp"

namespace Poseidon {

namespace Http {
	UpgradedLowLevelSessionBase::UpgradedLowLevelSessionBase(const boost::shared_ptr<Session> &parent, std::string uri)
		: m_parent(parent), m_uri(STD_MOVE(uri))
	{
	}

	void UpgradedLowLevelSessionBase::onReadHup() NOEXCEPT {
	}
	void UpgradedLowLevelSessionBase::onWriteHup() NOEXCEPT {
	}
	void UpgradedLowLevelSessionBase::onClose(int errCode) NOEXCEPT {
		(void)errCode;
	}

	void UpgradedLowLevelSessionBase::onInit(RequestHeaders requestHeaders, StreamBuffer entity){
		(void)requestHeaders;
		(void)entity;
	}

	bool UpgradedLowLevelSessionBase::send(StreamBuffer buffer){
		const AUTO(parent, getParent());
		if(!parent){
			return false;
		}
		return static_cast<TcpSessionBase &>(*parent).send(STD_MOVE(buffer));
	}

	bool UpgradedLowLevelSessionBase::hasBeenShutdownRead() const NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return true;
		}
		return static_cast<const TcpSessionBase &>(*parent).hasBeenShutdownRead();
	}
	bool UpgradedLowLevelSessionBase::shutdownRead() NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return false;
		}
		return static_cast<TcpSessionBase &>(*parent).shutdownRead();
	}
	bool UpgradedLowLevelSessionBase::hasBeenShutdownWrite() const NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return true;
		}
		return static_cast<const TcpSessionBase &>(*parent).hasBeenShutdownWrite();
	}
	bool UpgradedLowLevelSessionBase::shutdownWrite() NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return false;
		}
		return static_cast<TcpSessionBase &>(*parent).shutdownWrite();
	}
	void UpgradedLowLevelSessionBase::forceShutdown() NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return;
		}
		static_cast<TcpSessionBase &>(*parent).forceShutdown();
	}

	void UpgradedLowLevelSessionBase::setTimeout(boost::uint64_t timeout){
		getSafeParent()->setTimeout(timeout);
	}
}

}
