// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MONGODB_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MONGODB_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace Poseidon {

namespace MongoDb {
	class ObjectBase;
	class Connection;
	class BsonBuilder;
}

class JobPromise;

class MongoDbDaemon {
private:
	MongoDbDaemon();

public:
	typedef boost::function<void (const boost::shared_ptr<MongoDb::Connection> &)> QueryCallback;

	static void start();
	static void stop();

	// 同步接口。
	static boost::shared_ptr<MongoDb::Connection> create_connection(bool from_slave = false);

	static void wait_for_all_async_operations();

	// 异步接口。
	static boost::shared_ptr<const JobPromise> enqueue_for_saving(
		boost::shared_ptr<const MongoDb::ObjectBase> object, bool to_replace, bool urgent);
	static boost::shared_ptr<const JobPromise> enqueue_for_loading(
		boost::shared_ptr<MongoDb::ObjectBase> object, MongoDb::BsonBuilder query);
	static boost::shared_ptr<const JobPromise> enqueue_for_deleting(
		const char *collection, MongoDb::BsonBuilder query, bool delete_all);
	static boost::shared_ptr<const JobPromise> enqueue_for_batch_loading(
		QueryCallback callback, const char *collection, MongoDb::BsonBuilder query, boost::uint32_t begin, boost::uint32_t limit);

	static void enqueue_for_low_level_access(boost::shared_ptr<JobPromise> promise,
		QueryCallback callback, const char *collection, bool from_slave = false);

	static boost::shared_ptr<const JobPromise> enqueue_for_waiting_for_all_async_operations();
};

}

#endif
