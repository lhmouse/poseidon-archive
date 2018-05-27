// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "upgraded_session_base.hpp"
#include "../tcp_session_base.hpp"
#include "../ip_port.hpp"
#include "../exception.hpp"
#include "../profiler.hpp"

namespace Poseidon {
namespace Http {

Upgraded_session_base::Upgraded_session_base(const boost::shared_ptr<Tcp_session_base> &parent)
	: m_parent(parent)
{
	//
}
Upgraded_session_base::~Upgraded_session_base(){
	//
}

void Upgraded_session_base::on_shutdown_timer(boost::uint64_t /*now*/){
	//
}

bool Upgraded_session_base::has_been_shutdown_read() const NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return true;
	}
	return parent->has_been_shutdown_read();
}
bool Upgraded_session_base::shutdown_read() NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return false;
	}
	return parent->shutdown_read();
}
bool Upgraded_session_base::has_been_shutdown_write() const NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return true;
	}
	return parent->has_been_shutdown_write();
}
bool Upgraded_session_base::shutdown_write() NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return false;
	}
	return parent->shutdown_write();
}
void Upgraded_session_base::force_shutdown() NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return;
	}
	parent->force_shutdown();
}

const Ip_port &Upgraded_session_base::get_remote_info() const NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return unknown_ip_port();
	}
	return parent->get_remote_info();
}
const Ip_port &Upgraded_session_base::get_local_info() const NOEXCEPT {
	const AUTO(parent, get_parent());
	if(!parent){
		return unknown_ip_port();
	}
	return parent->get_local_info();
}

bool Upgraded_session_base::is_throttled() const {
	const AUTO(parent, get_parent());
	if(!parent){
		return false;
	}
	return parent->is_throttled();
}

void Upgraded_session_base::set_no_delay(bool enabled){
	const AUTO(parent, get_parent());
	POSEIDON_THROW_UNLESS(parent, Basic_exception, Rcnts::view("Parent session is gone"));
	parent->set_no_delay(enabled);
}
void Upgraded_session_base::set_timeout(boost::uint64_t timeout){
	const AUTO(parent, get_parent());
	POSEIDON_THROW_UNLESS(parent, Basic_exception, Rcnts::view("Parent session is gone"));
	parent->set_timeout(timeout);
}

bool Upgraded_session_base::send(Stream_buffer buffer){
	const AUTO(parent, get_parent());
	if(!parent){
		return false;
	}
	return parent->send(STD_MOVE(buffer));
}

}
}
