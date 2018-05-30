// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "object_base.hpp"
#include "connection.hpp"
#include "../singletons/mongodb_daemon.hpp"
#include "../atomic.hpp"
#include "../log.hpp"

namespace Poseidon {
namespace Mongodb {

template class Object_base::Field<bool>;
template class Object_base::Field<boost::int64_t>;
template class Object_base::Field<boost::uint64_t>;
template class Object_base::Field<double>;
template class Object_base::Field<std::string>;
template class Object_base::Field<Uuid>;
template class Object_base::Field<Stream_buffer>;

Object_base::~Object_base(){
	//
}

bool Object_base::is_auto_saving_enabled() const NOEXCEPT {
	return atomic_load(m_auto_saves, memory_order_consume);
}
void Object_base::enable_auto_saving() const NOEXCEPT {
	atomic_store(m_auto_saves, true, memory_order_release);
}
void Object_base::disable_auto_saving() const NOEXCEPT {
	atomic_store(m_auto_saves, false, memory_order_release);
}

bool Object_base::invalidate() const NOEXCEPT
try {
	if(!is_auto_saving_enabled()){
		return false;
	}
	Mongodb_daemon::enqueue_for_saving(virtual_shared_from_this<Object_base>(), true, false);
	return true;
} catch(std::exception &e){
	POSEIDON_LOG_ERROR("std::exception thrown: what = ", e.what());
	return false;
} catch(...){
	POSEIDON_LOG_ERROR("Unknown exception thrown.");
	return false;
}

void * Object_base::get_combined_write_stamp() const NOEXCEPT {
	return atomic_load(m_combined_write_stamp, memory_order_consume);
}
void Object_base::set_combined_write_stamp(void *stamp) const NOEXCEPT {
	atomic_store(m_combined_write_stamp, stamp, memory_order_release);
}

// Non-member functions.
void enqueue_for_saving(const boost::shared_ptr<Object_base> &obj){
	Mongodb_daemon::enqueue_for_saving(obj, true, true);
}

}
}
