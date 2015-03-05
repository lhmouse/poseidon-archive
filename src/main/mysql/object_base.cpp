// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "object_base.hpp"
#include "../singletons/mysql_daemon.hpp"

namespace Poseidon {

namespace MySql {
	void ObjectBase::batchLoad(boost::shared_ptr<ObjectBase> (*factory)(),
		const char *tableHint, std::string query,
		BatchAsyncLoadCallback callback, ExceptionCallback except)
	{
		Daemon::enqueueForBatchLoading(
			factory, tableHint, STD_MOVE(query), STD_MOVE(callback), STD_MOVE(except));
	}

	ObjectBase::ObjectBase()
		: m_autoSaves(false), m_context()
	{
	}
	ObjectBase::~ObjectBase(){
	}

	bool ObjectBase::invalidate() const NOEXCEPT {
		try {
			if(isAutoSavingEnabled()){
				Daemon::enqueueForSaving(
					virtualSharedFromThis<ObjectBase>(), true, AsyncSaveCallback(), ExceptionCallback());
				return true;
			}
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown while enqueueing MySQL operation: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown while enqueueing MySQL operation");
		}
		return false;
	}

	void ObjectBase::asyncSave(bool toReplace, AsyncSaveCallback callback, ExceptionCallback except) const {
		enableAutoSaving();
		Daemon::enqueueForSaving(virtualSharedFromThis<ObjectBase>(),
			toReplace, STD_MOVE(callback), STD_MOVE(except));
	}
	void ObjectBase::asyncLoad(std::string query, AsyncLoadCallback callback, ExceptionCallback except){
		disableAutoSaving();
		Daemon::enqueueForLoading(
			virtualSharedFromThis<ObjectBase>(), STD_MOVE(query), STD_MOVE(callback), STD_MOVE(except));
	}
}

}
