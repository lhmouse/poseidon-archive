// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>
#include "../mysql/callbacks.hpp"

namespace Poseidon {

struct MySqlSnapshotItem {
	unsigned thread;
	const char *table;
	unsigned long long usTotal;
};

class MySqlObjectBase;

struct MySqlDaemon {
	static void start();
	static void stop();

	static std::vector<MySqlSnapshotItem> snapshot();

	static void waitForAllAsyncOperations();

	static void enqueueForSaving(boost::shared_ptr<const MySqlObjectBase> object, bool toReplace,
		MySqlAsyncSaveCallback callback, MySqlExceptionCallback except);
	static void enqueueForLoading(boost::shared_ptr<MySqlObjectBase> object, std::string query,
		MySqlAsyncLoadCallback callback, MySqlExceptionCallback except);
	static void enqueueForBatchLoading(boost::shared_ptr<MySqlObjectBase> (*factory)(),
		const char *tableHint, std::string query,
		MySqlBatchAsyncLoadCallback callback, MySqlExceptionCallback except);

private:
	MySqlDaemon();
};

}

#endif
