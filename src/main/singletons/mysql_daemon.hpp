// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>
#include "../mysql/callbacks.hpp"

namespace Poseidon {

struct MySqlSnapshotItem {
	unsigned index;
	unsigned long pendingOperations;
	unsigned long long usIdle;
	unsigned long long usWorking;
};

class MySqlObjectBase;

struct MySqlDaemon {
	static void start();
	static void stop();

	static std::vector<MySqlSnapshotItem> snapshot();

	static void waitForAllAsyncOperations();

	static void pendForSaving(boost::shared_ptr<const MySqlObjectBase> object, bool replaces,
		MySqlAsyncSaveCallback callback);
	static void pendForLoading(boost::shared_ptr<MySqlObjectBase> object, std::string query,
		MySqlAsyncLoadCallback callback);
	static void pendForBatchAsyncLoading(const char *tableHint, std::string query,
		boost::shared_ptr<MySqlObjectBase> (*factory)(), MySqlBatchAsyncLoadCallback callback);

private:
	MySqlDaemon();
};

}

#endif
