// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "object_base.hpp"
#include "exception.hpp"
#include "../singletons/mysql_daemon.hpp"
#include "../singletons/job_dispatcher.hpp"

namespace Poseidon {

namespace MySql {
	void ObjectBase::batchLoad(boost::shared_ptr<ObjectBase> (*factory)(), const char *tableHint, std::string query){
		const AUTO(promise, MySqlDaemon::enqueueForBatchLoading(factory, tableHint, STD_MOVE(query)));
		JobDispatcher::yield(boost::bind(&Promise::isSatisfied, promise));
		promise->checkAndRethrow();
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
				MySqlDaemon::enqueueForSaving(virtualSharedFromThis<ObjectBase>(), true, false);
				return true;
			}
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown while enqueueing MySQL operation: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown while enqueueing MySQL operation");
		}
		return false;
	}

	void ObjectBase::asyncSave(bool toReplace, bool urgent) const {
		const AUTO(promise, MySqlDaemon::enqueueForSaving(virtualSharedFromThis<ObjectBase>(), toReplace, urgent));
		JobDispatcher::yield(boost::bind(&Promise::isSatisfied, promise));
		promise->checkAndRethrow();
		enableAutoSaving();
	}
	void ObjectBase::asyncLoad(std::string query){
		disableAutoSaving();

		const AUTO(promise, MySqlDaemon::enqueueForLoading(virtualSharedFromThis<ObjectBase>(), STD_MOVE(query)));
		JobDispatcher::yield(boost::bind(&Promise::isSatisfied, promise));
		promise->checkAndRethrow();
		enableAutoSaving();
	}
}

}
