// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "object_base.hpp"
#include "../singletons/mysql_daemon.hpp"

namespace Poseidon {

void MySqlObjectBase::batchLoad(
	boost::shared_ptr<MySqlObjectBase> (*factory)(), const char *tableHint, std::string query,
	MySqlBatchAsyncLoadCallback callback, MySqlExceptionCallback except)
{
	MySqlDaemon::enqueueForBatchLoading(
		factory, tableHint, STD_MOVE(query), STD_MOVE(callback), STD_MOVE(except));
}

MySqlObjectBase::MySqlObjectBase()
	: m_autoSaves(false), m_context()
{
}
MySqlObjectBase::~MySqlObjectBase(){
}

bool MySqlObjectBase::invalidate() const NOEXCEPT {
	try {
		if(isAutoSavingEnabled()){
			MySqlDaemon::enqueueForSaving(virtualSharedFromThis<MySqlObjectBase>(),
				true, MySqlAsyncSaveCallback(), MySqlExceptionCallback());
			return true;
		}
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("std::exception thrown while enqueueing MySQL operation: what = ", e.what());
	} catch(...){
		LOG_POSEIDON_ERROR("Unknown exception thrown while enqueueing MySQL operation");
	}
	return false;
}

void MySqlObjectBase::asyncSave(bool toReplace,
	MySqlAsyncSaveCallback callback, MySqlExceptionCallback except) const
{
	enableAutoSaving();
	MySqlDaemon::enqueueForSaving(virtualSharedFromThis<MySqlObjectBase>(),
		toReplace, STD_MOVE(callback), STD_MOVE(except));
}
void MySqlObjectBase::asyncLoad(std::string query,
	MySqlAsyncLoadCallback callback, MySqlExceptionCallback except)
{
	disableAutoSaving();
	MySqlDaemon::enqueueForLoading(virtualSharedFromThis<MySqlObjectBase>(),
		STD_MOVE(query), STD_MOVE(callback), STD_MOVE(except));
}

}
