// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "object_base.hpp"
#include "connection.hpp"
#include "../singletons/mysql_daemon.hpp"
#include "../atomic.hpp"

namespace Poseidon {
namespace MySql {

template class ObjectBase::Field<bool>;
template class ObjectBase::Field<boost::int64_t>;
template class ObjectBase::Field<boost::uint64_t>;
template class ObjectBase::Field<double>;
template class ObjectBase::Field<std::string>;
template class ObjectBase::Field<Uuid>;
template class ObjectBase::Field<std::basic_string<unsigned char> >;

ObjectBase::~ObjectBase(){
	//
}

bool ObjectBase::is_auto_saving_enabled() const NOEXCEPT {
	return atomic_load(m_auto_saves, memory_order_consume);
}
void ObjectBase::enable_auto_saving() const NOEXCEPT {
	atomic_store(m_auto_saves, true, memory_order_release);
}
void ObjectBase::disable_auto_saving() const NOEXCEPT {
	atomic_store(m_auto_saves, false, memory_order_release);
}

bool ObjectBase::invalidate() const NOEXCEPT
try {
	if(!is_auto_saving_enabled()){
		return false;
	}
	MySqlDaemon::enqueue_for_saving(virtual_shared_from_this<ObjectBase>(), true, false);
	return true;
} catch(std::exception &e){
	LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
	return false;
} catch(...){
	LOG_POSEIDON_ERROR("Unknown exception thrown.");
	return false;
}

void *ObjectBase::get_combined_write_stamp() const NOEXCEPT {
	return atomic_load(m_combined_write_stamp, memory_order_consume);
}
void ObjectBase::set_combined_write_stamp(void *stamp) const NOEXCEPT {
	atomic_store(m_combined_write_stamp, stamp, memory_order_release);
}

// Non-member functions.
void enqueue_for_saving(const boost::shared_ptr<ObjectBase> &obj){
	MySqlDaemon::enqueue_for_saving(obj, true, true);
}

}
}
