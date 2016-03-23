// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "object_base.hpp"
#include "connection.hpp"
#include "exception.hpp"
#include "../singletons/mysql_daemon.hpp"
#include "../atomic.hpp"

namespace Poseidon {

namespace MySql {
	ObjectBase::ObjectBase()
		: m_auto_saves(false), m_combined_write_stamp(NULLPTR)
	{
	}
	ObjectBase::~ObjectBase(){
	}

	bool ObjectBase::is_auto_saving_enabled() const {
		return atomic_load(m_auto_saves, ATOMIC_CONSUME);
	}
	void ObjectBase::enable_auto_saving() const {
		atomic_store(m_auto_saves, true, ATOMIC_RELEASE);
	}
	void ObjectBase::disable_auto_saving() const {
		atomic_store(m_auto_saves, false, ATOMIC_RELEASE);
	}

	bool ObjectBase::invalidate() const NOEXCEPT {
		if(!is_auto_saving_enabled()){
			return false;
		}
		try {
			async_save(true, false);
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception: what = ", e.what());
			return false;
		}
		return true;
	}

	void *ObjectBase::get_combined_write_stamp() const {
		return atomic_load(m_combined_write_stamp, ATOMIC_CONSUME);
	}
	void ObjectBase::set_combined_write_stamp(void *stamp) const {
		atomic_store(m_combined_write_stamp, stamp, ATOMIC_RELEASE);
	}
	void ObjectBase::async_save(bool to_replace, bool urgent) const {
		enable_auto_saving();
		MySqlDaemon::enqueue_for_saving(virtual_shared_from_this<ObjectBase>(), to_replace, urgent);
	}
}

}
