// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "object_base.hpp"
#include "../singletons/mysql_daemon.hpp"
using namespace Poseidon;

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
	MySqlDaemon::pendForSaving(
		virtualSharedFromThis<MySqlObjectBase>());
}
void MySqlObjectBase::asyncLoad(std::string filter, MySqlAsyncLoadCallback callback){
	disableAutoSaving();
	MySqlDaemon::pendForLoading(
		virtualSharedFromThis<MySqlObjectBase>(), STD_MOVE(filter), STD_MOVE(callback));
}
