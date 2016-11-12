// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "upgraded_client_base.hpp"
#include "low_level_client.hpp"

namespace Poseidon {

namespace Http {
	UpgradedClientBase::UpgradedClientBase(const boost::shared_ptr<LowLevelClient> &parent)
		: m_parent(parent)
	{
	}
	UpgradedClientBase::~UpgradedClientBase(){
	}

	void UpgradedClientBase::on_read_hup() NOEXCEPT {
	}
	void UpgradedClientBase::on_close(int err_code) NOEXCEPT {
		(void)err_code;
	}

	bool UpgradedClientBase::send(StreamBuffer buffer){
		const AUTO(parent, get_parent());
		if(!parent){
			return false;
		}
		return parent->TcpClientBase::send(STD_MOVE(buffer));
	}

	bool UpgradedClientBase::has_been_shutdown_read() const NOEXCEPT {
		const AUTO(parent, get_parent());
		if(!parent){
			return true;
		}
		return parent->TcpClientBase::has_been_shutdown_read();
	}
	bool UpgradedClientBase::shutdown_read() NOEXCEPT {
		const AUTO(parent, get_parent());
		if(!parent){
			return false;
		}
		return parent->TcpClientBase::shutdown_read();
	}
	bool UpgradedClientBase::has_been_shutdown_write() const NOEXCEPT {
		const AUTO(parent, get_parent());
		if(!parent){
			return true;
		}
		return parent->TcpClientBase::has_been_shutdown_write();
	}
	bool UpgradedClientBase::shutdown_write() NOEXCEPT {
		const AUTO(parent, get_parent());
		if(!parent){
			return false;
		}
		return parent->TcpClientBase::shutdown_write();
	}
	void UpgradedClientBase::force_shutdown() NOEXCEPT {
		const AUTO(parent, get_parent());
		if(!parent){
			return;
		}
		parent->TcpClientBase::force_shutdown();
	}
}

}
