// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "upgraded_session_base.hpp"
#include "../tcp_session_base.hpp"
#include "../ip_port.hpp"
#include "../exception.hpp"

namespace Poseidon {
namespace Http {

UpgradedSessionBase::UpgradedSessionBase(const boost::shared_ptr<TcpSessionBase> &parent)
	: m_parent(parent)
{ }
UpgradedSessionBase::~UpgradedSessionBase(){ }

void UpgradedSessionBase::on_shutdown_timer(boost::uint64_t now){
	(void)now;
}

bool UpgradedSessionBase::has_been_shutdown_read() const NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return true;
	}
	return parent->has_been_shutdown_read();
}
bool UpgradedSessionBase::shutdown_read() NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return false;
	}
	return parent->shutdown_read();
}
bool UpgradedSessionBase::has_been_shutdown_write() const NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return true;
	}
	return parent->has_been_shutdown_write();
}
bool UpgradedSessionBase::shutdown_write() NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return false;
	}
	return parent->shutdown_write();
}
void UpgradedSessionBase::force_shutdown() NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return;
	}
	parent->force_shutdown();
}

const IpPort &UpgradedSessionBase::get_remote_info() const NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return unknown_ip_port();
	}
	return parent->get_remote_info();
}
const IpPort &UpgradedSessionBase::get_local_info() const NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return unknown_ip_port();
	}
	return parent->get_local_info();
}

bool UpgradedSessionBase::is_throttled() const {
	const AUTO(parent, get_parent());
	if(!parent){
		return false;
	}
	return parent->is_throttled();
}

void UpgradedSessionBase::set_no_delay(bool enabled){
	const AUTO(parent, get_parent());
	DEBUG_THROW_UNLESS(parent, BasicException, sslit("Parent session is gone"));
	parent->set_no_delay(enabled);
}
void UpgradedSessionBase::set_timeout(boost::uint64_t timeout){
	const AUTO(parent, get_parent());
	DEBUG_THROW_UNLESS(parent, BasicException, sslit("Parent session is gone"));
	parent->set_timeout(timeout);
}

bool UpgradedSessionBase::send(StreamBuffer buffer){
	const AUTO(parent, get_parent());
	if(!parent){
		return false;
	}
	return parent->send(STD_MOVE(buffer));
}

}
}
