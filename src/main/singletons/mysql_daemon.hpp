// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>
#include "../mysql/callbacks.hpp"

namespace Poseidon {

namespace MySql {
	class ObjectBase;
	class Connection;

	struct SnapshotItem {
		unsigned thread;
		const char *table;
		unsigned long long usTotal;
	};

	struct Daemon {
		static void start();
		static void stop();

		// 同步接口。
		static boost::shared_ptr<Connection> createConnection();

		// 异步接口。
		static std::vector<SnapshotItem> snapshot();

		static void waitForAllAsyncOperations();

		static void enqueueForSaving(boost::shared_ptr<const ObjectBase> object, bool toReplace,
			AsyncSaveCallback callback, ExceptionCallback except);
		static void enqueueForLoading(boost::shared_ptr<ObjectBase> object, std::string query,
			AsyncLoadCallback callback, ExceptionCallback except);
		static void enqueueForBatchLoading(boost::shared_ptr<ObjectBase> (*factory)(),
			const char *tableHint, std::string query,
			BatchAsyncLoadCallback callback, ExceptionCallback except);

	private:
		Daemon();
	};
}

typedef MySql::Daemon MySqlDaemon;

}

#endif
