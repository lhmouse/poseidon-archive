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
	UpgradedSessionBase::~UpgradedSessionBase(){
	}

	void UpgradedSessionBase::onConnect(){
	}
	void UpgradedSessionBase::onReadHup() NOEXCEPT {
	}
	void UpgradedSessionBase::onClose(int errCode) NOEXCEPT {
		(void)errCode;
	}

	bool UpgradedSessionBase::send(StreamBuffer buffer){
		const AUTO(parent, getParent());
		if(!parent){
			return false;
		}
		return parent->TcpSessionBase::send(STD_MOVE(buffer));
	}

	bool UpgradedSessionBase::hasBeenShutdownRead() const NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return true;
		}
		return parent->TcpSessionBase::hasBeenShutdownRead();
	}
	bool UpgradedSessionBase::shutdownRead() NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return false;
		}
		return parent->TcpSessionBase::shutdownRead();
	}
	bool UpgradedSessionBase::hasBeenShutdownWrite() const NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return true;
		}
		return parent->TcpSessionBase::hasBeenShutdownWrite();
	}
	bool UpgradedSessionBase::shutdownWrite() NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return false;
		}
		return parent->TcpSessionBase::shutdownWrite();
	}
	void UpgradedSessionBase::forceShutdown() NOEXCEPT {
		const AUTO(parent, getParent());
		if(!parent){
			return;
		}
		parent->TcpSessionBase::forceShutdown();
	}

	void UpgradedSessionBase::setTimeout(boost::uint64_t timeout){
		const AUTO(parent, getParent());
		if(!parent){
			return;
		}
		parent->TcpSessionBase::setTimeout(timeout);
	}
}

}
