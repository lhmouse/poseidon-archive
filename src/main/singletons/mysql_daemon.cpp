// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "mysql_daemon.hpp"
#include <boost/thread.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/bind.hpp>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include "main_config.hpp"
#include "../mysql/object_base.hpp"
#define POSEIDON_MYSQL_OBJECT_IMPL_
#include "../mysql/object_impl.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../utilities.hpp"
using namespace Poseidon;

namespace {

std::string g_mySqlServer			= "tcp://localhost:3306";
std::string g_mySqlUsername			= "root";
std::string g_mySqlPassword			= "root";
std::string g_mySqlSchema			= "test";

std::size_t g_mySqlMaxThreads		= 3;
std::size_t g_mySqlSaveDelay		= 5000;
std::size_t g_mySqlMaxReconnDelay	= 60000;
std::size_t g_mySqlRetryCount		= 3;

class AsyncLoadCallbackJob : public JobBase {
private:
	MySqlAsyncLoadCallback m_callback;
	boost::shared_ptr<MySqlObjectBase> m_object;

public:
	explicit AsyncLoadCallbackJob(Move<MySqlAsyncLoadCallback> callback,
		boost::shared_ptr<MySqlObjectBase> object)
	{
		callback.swap(m_callback);
		object.swap(m_object);
	}

protected:
	void perform(){
		PROFILE_ME;

		m_callback(STD_MOVE(m_object));
	}
};

struct AsyncSaveItem {
	boost::shared_ptr<const MySqlObjectBase> object;
	unsigned long long timeStamp;

	void swap(AsyncSaveItem &rhs) NOEXCEPT {
		object.swap(rhs.object);
		std::swap(timeStamp, rhs.timeStamp);
	}
};

struct AsyncLoadItem {
	boost::shared_ptr<MySqlObjectBase> object;
	std::string filter;
	MySqlAsyncLoadCallback callback;

	void swap(AsyncLoadItem &rhs) NOEXCEPT {
		object.swap(rhs.object);
		filter.swap(rhs.filter);
		callback.swap(rhs.callback);
	}
};

struct MySqlThreadContext {
	boost::thread thread;

	boost::mutex mutex;
	std::list<AsyncSaveItem> saveQueue;
	std::list<AsyncLoadItem> loadQueue;

	volatile std::size_t waiting;
	boost::condition_variable newObjectAvail;
	boost::condition_variable queueEmpty;

	volatile bool connected;

	MySqlThreadContext()
		: waiting(0), connected(false)
	{
	}
};

volatile bool g_running = false;
std::vector<boost::shared_ptr<MySqlThreadContext> > g_threads;

boost::mutex g_mutex;
std::list<AsyncSaveItem> g_savePool;
std::list<AsyncLoadItem> g_loadPool;
// 根据表名分给不同的线程，轮转调度。
std::size_t g_counter;
std::map<const char *, boost::weak_ptr<MySqlThreadContext> > g_assignments;

bool getMySqlConnection(boost::scoped_ptr<sql::Connection> &connection){
	LOG_POSEIDON_INFO("Connecting to MySQL server: ", g_mySqlServer);
	try {
		connection.reset(::get_driver_instance()->connect(
			g_mySqlServer, g_mySqlUsername, g_mySqlPassword));
		connection->setSchema(g_mySqlSchema);
		LOG_POSEIDON_INFO("Successfully connected to MySQL server.");
		return true;
	} catch(sql::SQLException &e){
		LOG_POSEIDON_ERROR("Error connecting to MySQL server: code = ", e.getErrorCode(),
			", state = ", e.getSQLState(), ", what = ", e.what());
		return false;
	}
}

void daemonLoop(const boost::shared_ptr<MySqlThreadContext> &ctx){
	boost::scoped_ptr<sql::Connection> connection;
	if(!getMySqlConnection(connection)){
		LOG_POSEIDON_FATAL("Failed to connect MySQL server. Bailing out.");
		std::abort();
	} else {
		const boost::mutex::scoped_lock lock(ctx->mutex);
		atomicStore(ctx->connected, true);
		ctx->queueEmpty.notify_all();
	}

	std::size_t reconnectDelay = 0;

	AsyncSaveItem retryAsi;
	AsyncLoadItem retryAli;
	std::size_t retryCount = 0;

	for(;;){
		bool discardConnection = false;
		AsyncSaveItem asi;
		AsyncLoadItem ali;

		try {
			if(!connection){
				LOG_POSEIDON_WARN("Lost connection to MySQL server. Reconnecting...");

				if(reconnectDelay == 0){
					reconnectDelay = 1;
				} else {
					LOG_POSEIDON_INFO("Will retry after ", reconnectDelay, " milliseconds.");

					boost::mutex::scoped_lock lock(ctx->mutex);
					ctx->newObjectAvail.timed_wait(lock, boost::posix_time::milliseconds(reconnectDelay));

					reconnectDelay <<= 1;
					if(reconnectDelay > g_mySqlMaxReconnDelay){
						reconnectDelay = g_mySqlMaxReconnDelay;
					}
				}
				if(!getMySqlConnection(connection)){
					if(!atomicLoad(g_running)){
						LOG_POSEIDON_WARN("Shutting down...");
						break;
					}
					continue;
				}
				reconnectDelay = 0;
			}

			if(retryCount == 0){
				boost::mutex::scoped_lock lock(ctx->mutex);
				for(;;){
					bool empty = true;
					if(!ctx->saveQueue.empty()){
						empty = false;
						AUTO_REF(head, ctx->saveQueue.front());
						if((atomicLoad(ctx->waiting) == 0) && (head.timeStamp > getMonoClock())){
							goto skip;
						}
						if(MySqlObjectImpl::getContext(*head.object) != &head){
							AsyncSaveItem().swap(head);
						} else {
							asi.swap(head);
						}
						{
							const boost::mutex::scoped_lock poolLock(g_mutex);
							g_savePool.splice(g_savePool.begin(), ctx->saveQueue, ctx->saveQueue.begin());
						}
						if(!asi.object){
							goto skip;
						}
						break;
					}
				skip:
					if(!ctx->loadQueue.empty()){
						empty = false;
						ali.swap(ctx->loadQueue.front());
						{
							const boost::mutex::scoped_lock poolLock(g_mutex);
							g_loadPool.splice(g_loadPool.begin(), ctx->loadQueue, ctx->loadQueue.begin());
						}
						break;
					}

					if(empty){
						ctx->queueEmpty.notify_all();
						if(!atomicLoad(g_running)){
							break;
						}
					}
					ctx->newObjectAvail.timed_wait(lock, boost::posix_time::seconds(1));
				}
				if(!asi.object && !ali.object){
					break;
				}
			} else {
				LOG_POSEIDON_INFO("Retrying last failed operation, retryCount = ", retryCount);

				retryAsi.swap(asi);
				retryAli.swap(ali);
			}
			if(asi.object){
				asi.object->syncSave(connection.get());
				asi.object->enableAutoSaving();
			}
			if(ali.object){
				ali.object->syncLoad(connection.get(), ali.filter.c_str());
				ali.object->enableAutoSaving();

				boost::make_shared<AsyncLoadCallbackJob>(
					STD_MOVE(ali.callback), STD_MOVE(ali.object)
					)->pend();
			}
		} catch(sql::SQLException &e){
			LOG_POSEIDON_ERROR("SQLException thrown in MySQL daemon: code = ", e.getErrorCode(),
				", state = ", e.getSQLState(), ", what = ", e.what());
			discardConnection = true;

			bool retry = false;
			if(retryCount == 0){
				if(g_mySqlRetryCount != 0){
					retryCount = g_mySqlRetryCount;
					retry = true;
				}
			} else {
				if(--retryCount != 0){
					retry = true;
				}
			}
			if(retry){
				retryAsi.swap(asi);
				retryAli.swap(ali);
			} else {
				LOG_POSEIDON_WARN("Retry count drops to zero. Give up.");
			}
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown in MySQL daemon: what = ", e.what());
			discardConnection = true;
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown in MySQL daemon.");
			discardConnection = true;
		}

		if(discardConnection && connection){
			LOG_POSEIDON_INFO("The connection was left in an indeterminate state. Discard it.");
			connection.reset();
		}
	}

	if(!ctx->saveQueue.empty()){
		LOG_POSEIDON_ERROR("There are still ", ctx->saveQueue.size(), " object(s) in MySQL save queue");
		ctx->saveQueue.clear();
	}
	ctx->loadQueue.clear();
}

void threadProc(boost::weak_ptr<MySqlThreadContext> weakContext){
	const boost::shared_ptr<MySqlThreadContext> ctx(weakContext);

	PROFILE_ME;
	Logger::setThreadTag(" D  "); // Database
	LOG_POSEIDON_INFO("MySQL daemon started.");

	::get_driver_instance()->threadInit();
	daemonLoop(ctx);
	::get_driver_instance()->threadEnd();

	LOG_POSEIDON_INFO("MySQL daemon stopped.");
}

}

void MySqlDaemon::start(){
	if(atomicExchange(g_running, true) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON_INFO("Starting MySQL daemon...");

	AUTO_REF(conf, MainConfig::getConfigFile());

	conf.get(g_mySqlServer, "mysql_server");
	LOG_POSEIDON_DEBUG("MySQL server = ", g_mySqlServer);

	conf.get(g_mySqlUsername, "mysql_username");
	LOG_POSEIDON_DEBUG("MySQL username = ", g_mySqlUsername);

	conf.get(g_mySqlPassword, "mysql_password");
	LOG_POSEIDON_DEBUG("MySQL password = ", g_mySqlPassword);

	conf.get(g_mySqlSchema, "mysql_schema");
	LOG_POSEIDON_DEBUG("MySQL schema = ", g_mySqlSchema);

	conf.get(g_mySqlMaxThreads, "mysql_max_threads");
	LOG_POSEIDON_DEBUG("MySQL max threads = ", g_mySqlMaxThreads);

	conf.get(g_mySqlSaveDelay, "mysql_save_delay");
	LOG_POSEIDON_DEBUG("MySQL save delay = ", g_mySqlSaveDelay);

	conf.get(g_mySqlMaxReconnDelay, "mysql_max_reconn_delay");
	LOG_POSEIDON_DEBUG("MySQL max reconnect delay = ", g_mySqlMaxReconnDelay);

	conf.get(g_mySqlRetryCount, "mysql_retry_count");
	LOG_POSEIDON_DEBUG("MySQL retry count = ", g_mySqlRetryCount);

	g_threads.resize(std::max<std::size_t>(g_mySqlMaxThreads, 1));
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		AUTO(ctx, boost::make_shared<MySqlThreadContext>());
		boost::thread thread(boost::bind(&threadProc, boost::weak_ptr<MySqlThreadContext>(ctx)));
		ctx->thread.swap(thread);
		g_threads.at(i).swap(ctx);

		LOG_POSEIDON_INFO("Created MySQL thread ", i);
	}
}
void MySqlDaemon::stop(){
	LOG_POSEIDON_INFO("Stopping MySQL daemon...");

	atomicStore(g_running, false);
	waitForAllAsyncOperations();

	for(std::size_t i = 0; i < g_threads.size(); ++i){
		if(g_threads[i]->thread.joinable()){
			g_threads[i]->thread.join();
		}

		LOG_POSEIDON_INFO("MySQL thread ", i, " has terminated.");
	}
	g_threads.clear();
}

void MySqlDaemon::waitForAllAsyncOperations(){
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		atomicAdd(g_threads[i]->waiting, 1);
		try {
			LOG_POSEIDON_INFO("Waiting for all MySQL operations to complete on thread ", i);

			boost::mutex::scoped_lock lock(g_threads[i]->mutex);
			g_threads[i]->newObjectAvail.notify_all();
			while(!atomicLoad(g_threads[i]->connected) ||
				!(g_threads[i]->saveQueue.empty() && g_threads[i]->loadQueue.empty()))
			{
				g_threads[i]->queueEmpty.wait(lock);
			}
		} catch(...){
			LOG_POSEIDON_ERROR("Interrupted by exception.");
		}
		atomicSub(g_threads[i]->waiting, 1);
	}
}

void MySqlDaemon::pendForSaving(boost::shared_ptr<const MySqlObjectBase> object){
	std::list<AsyncSaveItem> saveItem;
	boost::shared_ptr<MySqlThreadContext> ctx;
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		if(g_savePool.empty()){
			saveItem.push_front(AsyncSaveItem());
		} else {
			saveItem.splice(saveItem.end(), g_savePool, g_savePool.begin());
		}

		const char *const table = object->getTableName();
		AUTO_REF(weakContext, g_assignments[table]);
		ctx = weakContext.lock();
		if(!ctx){
			const std::size_t index = ++g_counter % g_threads.size();
			ctx = g_threads[index];
			weakContext = ctx;

			LOG_POSEIDON_DEBUG("Assign MySQL table `", table, "` to thread ", index);
		}
	}

	AUTO_REF(asi, saveItem.back());
	asi.object.swap(object);
	asi.timeStamp = getMonoClock() + g_mySqlSaveDelay * 1000;
	MySqlObjectImpl::setContext(*asi.object, &asi);

	{
		const boost::mutex::scoped_lock lock(ctx->mutex);
		ctx->saveQueue.splice(ctx->saveQueue.end(), saveItem);
		ctx->newObjectAvail.notify_all();
	}
}
void MySqlDaemon::pendForLoading(boost::shared_ptr<MySqlObjectBase> object,
	std::string filter, MySqlAsyncLoadCallback callback)
{
	std::list<AsyncLoadItem> loadItem;
	boost::shared_ptr<MySqlThreadContext> ctx;
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		if(g_loadPool.empty()){
			loadItem.push_front(AsyncLoadItem());
		} else {
			loadItem.splice(loadItem.end(), g_loadPool, g_loadPool.begin());
		}

		const char *const table = object->getTableName();
		AUTO_REF(weakContext, g_assignments[table]);
		ctx = weakContext.lock();
		if(!ctx){
			const std::size_t index = ++g_counter % g_threads.size();
			ctx = g_threads[index];
			weakContext = ctx;

			LOG_POSEIDON_DEBUG("Assign MySQL table `", table, "` to thread ", index);
		}
	}

	AUTO_REF(ali, loadItem.back());
	ali.object.swap(object);
	ali.filter.swap(filter);
	ali.callback.swap(callback);

	{
		const boost::mutex::scoped_lock lock(ctx->mutex);
		ctx->loadQueue.splice(ctx->loadQueue.end(), loadItem);
		ctx->newObjectAvail.notify_all();
	}
}
