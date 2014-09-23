#include "../../precompiled.hpp"
#include "object_base.hpp"
#include <boost/bind.hpp>
#include "../log.hpp"
#include "../job_base.hpp"
#include "../utilities.hpp"
#include "../singletons/mysql_daemon.hpp"
using namespace Poseidon;

void MySqlObjectBase::asyncSave() const {
	MySqlDaemon::pendForSaving(virtualSharedFromThis<MySqlObjectBase>());
}
void MySqlObjectBase::asyncLoad(std::string filter, MySqlAsyncLoadCallback callback){
	MySqlDaemon::pendForLoading(virtualSharedFromThis<MySqlObjectBase>(),
		STD_MOVE(filter), STD_MOVE(callback));
}
