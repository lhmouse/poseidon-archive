// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "object_base.hpp"
#include "exception.hpp"
#include "../singletons/mysql_daemon.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../job_promise.hpp"
#include "../atomic.hpp"

namespace Poseidon {

namespace MySql {
	void ObjectBase::batch_load(std::vector<boost::shared_ptr<ObjectBase> > &ret,
		boost::shared_ptr<ObjectBase> (*factory)(), const char *table_hint, std::string query)
	{
		const AUTO(queue, boost::make_shared<std::deque<boost::shared_ptr<ObjectBase> > >());
		const AUTO(promise, MySqlDaemon::enqueue_for_batch_loading(queue, factory, table_hint, STD_MOVE(query)));
		JobDispatcher::yield(promise);

		ret.reserve(ret.size() + queue->size());
		ret.insert(ret.end(), queue->begin(), queue->end());
	}

	ObjectBase::ObjectBase()
		: m_auto_saves(false)
	{
	}
	ObjectBase::~ObjectBase(){
	}

	bool ObjectBase::invalidate() const NOEXCEPT {
		try {
			if(is_auto_saving_enabled()){
				async_save(true, false);
				return true;
			}
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception: what = ", e.what());
		}
		return false;
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

	void *ObjectBase::get_combined_write_stamp() const {
		return atomic_load(m_combined_write_stamp, ATOMIC_CONSUME);
	}
	void ObjectBase::set_combined_write_stamp(void *stamp) const {
		atomic_store(m_combined_write_stamp, stamp, ATOMIC_RELEASE);
	}

	void ObjectBase::save_and_wait(bool to_replace) const {
		const AUTO(promise, MySqlDaemon::enqueue_for_saving(virtual_shared_from_this<ObjectBase>(), to_replace, true));
		JobDispatcher::yield(promise);
		enable_auto_saving();
	}
	void ObjectBase::load_and_wait(std::string query){
		const AUTO(promise, MySqlDaemon::enqueue_for_loading(virtual_shared_from_this<ObjectBase>(), STD_MOVE(query)));
		JobDispatcher::yield(promise);
		enable_auto_saving();
	}
	void ObjectBase::async_save(bool to_replace, bool urgent) const {
		enable_auto_saving();
		MySqlDaemon::enqueue_for_saving(virtual_shared_from_this<ObjectBase>(), to_replace, urgent);
	}
}

}
