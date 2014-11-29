// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "object_base.hpp"
#include "../singletons/mysql_daemon.hpp"
using namespace Poseidon;

void MySqlObjectBase::batchAsyncLoad(const char *tableHint, std::string query,
	MySqlObjectFactoryCallback factory, MySqlBatchAsyncLoadCallback callback)
{
	MySqlDaemon::pendForBatchLoading(
		tableHint, STD_MOVE(query), STD_MOVE(factory), STD_MOVE(callback));
}

MySqlObjectBase::MySqlObjectBase()
	: m_autoSaves(false), m_context()
{
}
MySqlObjectBase::~MySqlObjectBase(){
}

void MySqlObjectBase::invalidate() const {
	if(!isAutoSavingEnabled()){
		return;
	}
	MySqlDaemon::pendForSaving(virtualSharedFromThis<MySqlObjectBase>());
}

void MySqlObjectBase::asyncSave() const {
	enableAutoSaving();
	MySqlDaemon::pendForSaving(virtualSharedFromThis<MySqlObjectBase>());
}
void MySqlObjectBase::asyncLoad(std::string query, MySqlAsyncLoadCallback callback){
	disableAutoSaving();
	MySqlDaemon::pendForLoading(virtualSharedFromThis<MySqlObjectBase>(),
		STD_MOVE(query), STD_MOVE(callback));
}
