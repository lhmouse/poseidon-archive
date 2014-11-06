#include "../precompiled.hpp"
#include "object_base.hpp"
#include "../singletons/mysql_daemon.hpp"
using namespace Poseidon;

MySqlObjectBase::MySqlObjectBase()
	: m_autoSaves(false), m_context()
{
}

void MySqlObjectBase::invalidate() const {
	if(!isAutoSavingEnabled()){
		return;
	}
	MySqlDaemon::pendForSaving(virtualSharedFromThis<MySqlObjectBase>());
}

void MySqlObjectBase::asyncSave() const {
	enableAutoSaving();
	MySqlDaemon::pendForSaving(
		virtualSharedFromThis<MySqlObjectBase>());
}
void MySqlObjectBase::asyncLoad(std::string filter, MySqlAsyncLoadCallback callback){
	disableAutoSaving();
	MySqlDaemon::pendForLoading(
		virtualSharedFromThis<MySqlObjectBase>(), STD_MOVE(filter), STD_MOVE(callback));
}
