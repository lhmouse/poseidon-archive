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
	void ObjectBase::batchLoad(std::vector<boost::shared_ptr<ObjectBase> > &ret,
		boost::shared_ptr<ObjectBase> (*factory)(), const char *tableHint, std::string query)
	{
		const AUTO(queue, boost::make_shared<std::deque<boost::shared_ptr<ObjectBase> > >());
		const AUTO(promise, MySqlDaemon::enqueueForBatchLoading(queue, factory, tableHint, STD_MOVE(query)));
		JobDispatcher::yield(promise);
		promise->checkAndRethrow();

		ret.reserve(ret.size() + queue->size());
		ret.insert(ret.end(), queue->begin(), queue->end());
	}

	ObjectBase::ObjectBase()
		: m_autoSaves(false)
	{
	}
	ObjectBase::~ObjectBase(){
	}

	bool ObjectBase::invalidate() const NOEXCEPT {
		try {
			if(isAutoSavingEnabled()){
				asyncSave(true, false);
				return true;
			}
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception: what = ", e.what());
		}
		return false;
	}

	bool ObjectBase::isAutoSavingEnabled() const {
		return atomicLoad(m_autoSaves, ATOMIC_CONSUME);
	}
	void ObjectBase::enableAutoSaving() const {
		atomicStore(m_autoSaves, true, ATOMIC_RELEASE);
	}
	void ObjectBase::disableAutoSaving() const {
		atomicStore(m_autoSaves, false, ATOMIC_RELEASE);
	}

	void *ObjectBase::getCombinedWriteStamp() const {
		return atomicLoad(m_combinedWriteStamp, ATOMIC_CONSUME);
	}
	void ObjectBase::setCombinedWriteStamp(void *stamp) const {
		atomicStore(m_combinedWriteStamp, stamp, ATOMIC_RELEASE);
	}

	void ObjectBase::syncSave(bool toReplace) const {
		const AUTO(promise, MySqlDaemon::enqueueForSaving(virtualSharedFromThis<ObjectBase>(), toReplace, true));
		JobDispatcher::yield(promise);
		promise->checkAndRethrow();
		enableAutoSaving();
	}
	void ObjectBase::syncLoad(std::string query){
		const AUTO(promise, MySqlDaemon::enqueueForLoading(virtualSharedFromThis<ObjectBase>(), STD_MOVE(query)));
		JobDispatcher::yield(promise);
		promise->checkAndRethrow();
		enableAutoSaving();
	}
	void ObjectBase::asyncSave(bool toReplace, bool urgent) const {
		enableAutoSaving();
		MySqlDaemon::enqueueForSaving(virtualSharedFromThis<ObjectBase>(), toReplace, urgent);
	}
}

}
