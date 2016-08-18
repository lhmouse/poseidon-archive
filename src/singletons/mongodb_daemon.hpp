// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MONGODB_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MONGODB_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace Poseidon {

namespace MongoDb {
	class BsonBuilder;
	class ObjectBase;
	class Connection;
}

class JobPromise;

struct MongoDbDaemon {
	typedef boost::function<void (const boost::shared_ptr<MongoDb::Connection> &)> ObjectFactory;

	static void start();
	static void stop();

	// 同步接口。
	static boost::shared_ptr<MongoDb::Connection> create_connection(bool from_slave = false);

	static void wait_for_all_async_operations();

	// 异步接口。
	// 以下第一个参数是出参。
	static boost::shared_ptr<const JobPromise> enqueue_for_saving(
		boost::shared_ptr<const MongoDb::ObjectBase> object, bool to_replace, bool urgent);
	static boost::shared_ptr<const JobPromise> enqueue_for_loading(
		boost::shared_ptr<MongoDb::ObjectBase> object, MongoDb::BsonBuilder query);
	static boost::shared_ptr<const JobPromise> enqueue_for_deleting(
		const char *collection, MongoDb::BsonBuilder query, bool delete_all);
	static boost::shared_ptr<const JobPromise> enqueue_for_batch_loading(
		ObjectFactory factory, const char *collection, MongoDb::BsonBuilder query, std::size_t begin = 0, std::size_t limit = 0x7FFFFFFF);

	static boost::shared_ptr<const JobPromise> enqueue_for_waiting_for_all_async_operations();

private:
	MongoDbDaemon();
};

}

#endif
