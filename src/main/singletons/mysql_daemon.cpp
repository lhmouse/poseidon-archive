// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "mysql_daemon.hpp"
#include "main_config.hpp"
#include <boost/thread/thread.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mysql/mysqld_error.h>
#include "../mysql/object_base.hpp"
#include "../mysql/exception.hpp"
#include "../mysql/thread_context.hpp"
#include "../mysql/connection.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../utilities.hpp"
using namespace Poseidon;

namespace {

std::string	g_mysqlServerAddr		= "localhost";
unsigned	g_mysqlServerPort		= 3306;
std::string	g_mysqlUsername			= "root";
std::string	g_mysqlPassword			= "root";
std::string	g_mysqlSchema			= "poseidon";
bool		g_mysqlUseSsl			= false;
std::string	g_mysqlCharset			= "utf8";

std::string	g_mysqlDumpDir			= "/usr/var/poseidon/sqldump";
std::size_t	g_mysqlMaxThreads		= 3;
std::size_t	g_mysqlSaveDelay		= 5000;
std::size_t	g_mysqlMaxReconnDelay	= 60000;
std::size_t	g_mysqlRetryCount		= 3;

// 转储文件，写在最前面。
boost::mutex g_dumpMutex;
UniqueFile g_dumpFile;

// 回调函数任务。
class SaveCallbackJob : public JobBase {
private:
	const MySqlAsyncSaveCallback m_callback;
	const bool m_succeeded;
	const boost::uint64_t m_autoId;
	const MySqlExceptionCallback m_except;

public:
	SaveCallbackJob(MySqlAsyncSaveCallback callback, bool succeeded, boost::uint64_t autoId,
		MySqlExceptionCallback except)
		: m_callback(STD_MOVE_IDN(callback)), m_succeeded(succeeded), m_autoId(autoId)
		, m_except(STD_MOVE_IDN(except))
	{
	}

private:
	void perform() OVERRIDE {
		PROFILE_ME;

		try {
			m_callback(m_succeeded, m_autoId);
		} catch(...){
			if(m_except){
				m_except();
			}
			throw;
		}
	}
};

class LoadCallbackJob : public JobBase {
private:
	const MySqlAsyncLoadCallback m_callback;
	const bool m_found;
	const MySqlExceptionCallback m_except;

public:
	LoadCallbackJob(MySqlAsyncLoadCallback callback, bool found,
		MySqlExceptionCallback except)
		: m_callback(STD_MOVE_IDN(callback)), m_found(found)
		, m_except(STD_MOVE_IDN(except))
	{
	}

private:
	void perform() OVERRIDE {
		PROFILE_ME;

		try {
			m_callback(m_found);
		} catch(...){
			if(m_except){
				m_except();
			}
			throw;
		}
	}
};

class BatchLoadCallbackJob : public JobBase {
private:
	const MySqlBatchAsyncLoadCallback m_callback;
	std::vector<boost::shared_ptr<MySqlObjectBase> > m_objects;
	const MySqlExceptionCallback m_except;

public:
	BatchLoadCallbackJob(MySqlBatchAsyncLoadCallback callback,
		std::vector<boost::shared_ptr<MySqlObjectBase> > objects,
		MySqlExceptionCallback except)
		: m_callback(STD_MOVE_IDN(callback)), m_objects(STD_MOVE(objects))
		, m_except(STD_MOVE_IDN(except))
	{
	}

private:
	void perform() OVERRIDE {
		PROFILE_ME;

		try {
			m_callback(STD_MOVE(m_objects));
		} catch(...){
			if(m_except){
				m_except();
			}
			throw;
		}
	}
};

// 数据库线程操作。
class OperationBase : NONCOPYABLE {
public:
	virtual ~OperationBase(){
	}

public:
	virtual bool check() const = 0;
	virtual void execute(std::string &query, MySqlConnection &conn) = 0;
};

class SaveOperation : public OperationBase {
private:
	const boost::uint64_t m_dueTime;

	const boost::shared_ptr<const MySqlObjectBase> m_object;
	const bool m_toReplace;

	MySqlAsyncSaveCallback m_callback;
	const MySqlExceptionCallback m_except;

public:
	SaveOperation(boost::uint64_t dueTime,
		boost::shared_ptr<const MySqlObjectBase> object, bool toReplace,
		MySqlAsyncSaveCallback callback, MySqlExceptionCallback except)
		: m_dueTime(dueTime)
		, m_object(STD_MOVE(object)), m_toReplace(toReplace)
		, m_callback(STD_MOVE_IDN(callback)), m_except(STD_MOVE_IDN(except))
	{
		m_object->setContext(this);
	}

public:
	bool check() const OVERRIDE {
		return m_dueTime <= getMonoClock();
	}
	void execute(std::string &query, MySqlConnection &conn){
		PROFILE_ME;

		try {
			if(m_object->getContext() != this){
				// 写入合并。
				return;
			}

			m_object->syncGenerateSql(query, m_toReplace);
			bool succeeded = false;
			try {
				LOG_POSEIDON_DEBUG("Executing SQL in ", m_object->getTableName(), ": ", query);
				conn.executeSql(query);
				succeeded = true;
			} catch(MySqlException &e){
				LOG_POSEIDON_DEBUG("MySqlException: code = ", e.code(), ", message = ", e.what());
				if(!m_callback || (e.code() != ER_DUP_ENTRY)){
					throw;
				}
			}

			if(m_callback){
				pendJob(boost::make_shared<SaveCallbackJob>(
					STD_MOVE(m_callback), succeeded, conn.getInsertId(), boost::ref(m_except)));
			}
		} catch(...){
			if(m_except){
				m_except();
			}
			throw;
		}
	}
};

class LoadOperation : public OperationBase {
private:
	const boost::shared_ptr<MySqlObjectBase> m_object;
	const std::string m_query;

	MySqlAsyncLoadCallback m_callback;
	const MySqlExceptionCallback m_except;

public:
	LoadOperation(boost::shared_ptr<MySqlObjectBase> object, std::string query,
		MySqlAsyncLoadCallback callback, MySqlExceptionCallback except)
		: m_object(STD_MOVE(object)), m_query(STD_MOVE(query))
		, m_callback(STD_MOVE_IDN(callback)), m_except(STD_MOVE_IDN(except))
	{
	}

public:
	bool check() const OVERRIDE {
		return true;
	}
	void execute(std::string &query, MySqlConnection &conn){
		PROFILE_ME;

		try {
			query = m_query;

			LOG_POSEIDON_INFO("MySQL load: table = ", m_object->getTableName(), ", query = ", query);
			conn.executeSql(query);
			const bool found = conn.fetchRow();
			if(found){
				m_object->syncFetch(conn);
				m_object->enableAutoSaving();
			}

			if(m_callback){
				pendJob(boost::make_shared<LoadCallbackJob>(
					STD_MOVE(m_callback), found, boost::ref(m_except)));
			}
		} catch(...){
			if(m_except){
				m_except();
			}
			throw;
		}
	}
};

class BatchLoadOperation : public OperationBase {
private:
	boost::shared_ptr<MySqlObjectBase> (*const m_factory)();
	const std::string m_query;

	MySqlBatchAsyncLoadCallback m_callback;
	const MySqlExceptionCallback m_except;

public:
	BatchLoadOperation(boost::shared_ptr<MySqlObjectBase> (*factory)(), std::string query,
		MySqlBatchAsyncLoadCallback callback, MySqlExceptionCallback except)
		: m_factory(factory), m_query(STD_MOVE(query))
		, m_callback(STD_MOVE_IDN(callback)), m_except(STD_MOVE_IDN(except))
	{
	}

private:
	bool check() const OVERRIDE {
		return true;
	}
	void execute(std::string &query, MySqlConnection &conn){
		PROFILE_ME;

		try {
			query = m_query;

			LOG_POSEIDON_INFO("MySQL batch load: query = ", query);
			conn.executeSql(query);

			std::vector<boost::shared_ptr<MySqlObjectBase> > objects;
			while(conn.fetchRow()){
				AUTO(object, (*m_factory)());
				object->syncFetch(conn);
				object->enableAutoSaving();
				objects.push_back(STD_MOVE(object));
			}

			if(m_callback){
				pendJob(boost::make_shared<BatchLoadCallbackJob>(
					STD_MOVE(m_callback), STD_MOVE(objects), boost::ref(m_except)));
			}
		} catch(...){
			if(m_except){
				m_except();
			}
			throw;
		}
	}
};

class MySqlThread : public boost::thread {
private:
	class WorkingTimeAccumulator : NONCOPYABLE {
	private:
		MySqlThread *const m_owner;

	public:
		explicit WorkingTimeAccumulator(MySqlThread *owner)
			: m_owner(owner)
		{
			m_owner->accumulateTimeIdle();
		}
		~WorkingTimeAccumulator(){
			m_owner->accumulateTimeWorking();
		}
	};

private:
	const std::size_t m_index;

	volatile bool m_running;

	mutable boost::mutex m_mutex;
	std::deque<boost::shared_ptr<OperationBase> > m_queue;
	volatile bool m_urgent; // 无视延迟写入，一次性处理队列中所有操作。
	boost::condition_variable m_newAvail;
	boost::condition_variable m_queueEmpty;

	// 性能统计。
	boost::uint64_t m_timeFlushed;
	volatile boost::uint64_t m_timeIdle;
	volatile boost::uint64_t m_timeWorking;

public:
	explicit MySqlThread(std::size_t index)
		: m_index(index)
		, m_running(true)
		, m_urgent(false)
		, m_timeFlushed(getMonoClock()), m_timeIdle(0), m_timeWorking(0)
	{
		boost::thread(&MySqlThread::loop, this).swap(*this);
	}

private:
	void accumulateTimeIdle() NOEXCEPT {
		const AUTO(now, getMonoClock());
		atomicAdd(m_timeIdle, now - m_timeFlushed, ATOMIC_RELAXED);
		m_timeFlushed = now;
	}
	void accumulateTimeWorking() NOEXCEPT {
		const AUTO(now, getMonoClock());
		atomicAdd(m_timeWorking, now - m_timeFlushed, ATOMIC_RELAXED);
		m_timeFlushed = now;
	}

	void loop();

public:
	void shutdown(){
		atomicStore(m_running, false, ATOMIC_RELEASE);
	}
	void join(){
		waitTillIdle();
		boost::thread::join();
	}

	std::size_t getPendingOperations() const {
		const boost::mutex::scoped_lock lock(m_mutex);
		return m_queue.size();
	}

	boost::uint64_t getTimeIdle() const {
		return atomicLoad(m_timeIdle, ATOMIC_RELAXED);
	}
	boost::uint64_t getTimeWorking() const {
		return atomicLoad(m_timeWorking, ATOMIC_RELAXED);
	}

	void addOperation(boost::shared_ptr<OperationBase> operation, bool urgent){
		const boost::mutex::scoped_lock lock(m_mutex);
		m_queue.push_back(STD_MOVE(operation));
		if(urgent){
			atomicStore(m_urgent, true, ATOMIC_RELEASE);
		}
		m_newAvail.notify_all();
	}
	void waitTillIdle(){
		atomicStore(m_urgent, true, ATOMIC_RELEASE); // 在获取互斥锁之前设置紧急状态。
		m_newAvail.notify_all();
		boost::mutex::scoped_lock lock(m_mutex);
		while(!m_queue.empty()){
			atomicStore(m_urgent, true, ATOMIC_RELEASE);
			m_newAvail.notify_all();
			m_queueEmpty.wait(lock);
		}
	}
};

void MySqlThread::loop(){
	PROFILE_ME;
	Logger::setThreadTag(" D  "); // Database
	LOG_POSEIDON_INFO("MySQL thread ", m_index, " started.");

	MySqlThreadContext context;

	boost::scoped_ptr<MySqlConnection> conn;
	std::size_t reconnectDelay = 0;

	boost::shared_ptr<OperationBase> operation;
	std::size_t retryCount = 0;

	for(;;){
		try {
			accumulateTimeIdle();

			if(!conn){
				LOG_POSEIDON_INFO("Connecting to MySQL server...");

				if(reconnectDelay == 0){
					reconnectDelay = 1;
				} else {
					LOG_POSEIDON_INFO("Will retry after ", reconnectDelay, " milliseconds.");

					boost::mutex::scoped_lock lock(m_mutex);
					m_newAvail.timed_wait(lock, boost::posix_time::milliseconds(reconnectDelay));

					reconnectDelay <<= 1;
					if(reconnectDelay > g_mysqlMaxReconnDelay){
						reconnectDelay = g_mysqlMaxReconnDelay;
					}
				}
				try {
					MySqlConnection::create(conn, context, g_mysqlServerAddr, g_mysqlServerPort,
						g_mysqlUsername, g_mysqlPassword, g_mysqlSchema, g_mysqlUseSsl, g_mysqlCharset);
				} catch(...){
					if(!atomicLoad(m_running, ATOMIC_ACQUIRE)){
						LOG_POSEIDON_WARN("Shutting down...");
						goto exit_loop;
					}
					throw;
				}

				LOG_POSEIDON_INFO("Successfully connected to MySQL server.");
				reconnectDelay = 0;
			}

			if(!operation){
				boost::mutex::scoped_lock lock(m_mutex);
				while(m_queue.empty()){
					m_queueEmpty.notify_all();
					atomicStore(m_urgent, false, ATOMIC_RELEASE);
					accumulateTimeIdle();

					if(!atomicLoad(m_running, ATOMIC_ACQUIRE)){
						goto exit_loop;
					}
					m_newAvail.timed_wait(lock, boost::posix_time::seconds(1));
				}
				operation.swap(m_queue.front());
				m_queue.pop_front();
			}
			if(!atomicLoad(m_urgent, ATOMIC_ACQUIRE) && !operation->check()){
				boost::mutex::scoped_lock lock(m_mutex);
				m_newAvail.timed_wait(lock, boost::posix_time::seconds(1));
				continue;
			}

			unsigned mysqlErrCode = 99999;
			std::string mysqlErrMsg;
			std::string query;
			try {
				try {
					const WorkingTimeAccumulator profiler(this);
					operation->execute(query, *conn);
				} catch(MySqlException &e){
					mysqlErrCode = e.code();
					mysqlErrMsg = e.what();
					throw;
				} catch(std::exception &e){
					mysqlErrMsg = e.what();
					throw;
				} catch(...){
					mysqlErrMsg = "Unknown exception";
					throw;
				}
			} catch(...){
				bool retry = true;
				if(retryCount == 0){
					if(g_mysqlRetryCount == 0){
						retry = false;
					} else {
						retryCount = g_mysqlRetryCount;
					}
				} else {
					if(--retryCount == 0){
						retry = false;
					}
				}
				if(!retry){
					LOG_POSEIDON_WARN("Retry count has dropped to zero. Give up.");
					operation.reset();

					char temp[32];
					unsigned len = std::sprintf(temp, "%05u", mysqlErrCode);
					std::string dump;
					dump.reserve(1024);
					dump.assign("-- Error code = ");
					dump.append(temp, len);
					dump.append(", Description = ");
					dump.append(mysqlErrMsg);
					dump.append("\n");
					dump.append(query);
					dump.append(";\n\n");
					{
						const boost::mutex::scoped_lock dumpLock(g_dumpMutex);
						std::size_t total = 0;
						do {
							::ssize_t written = ::write(
								g_dumpFile.get(), dump.data() + total, dump.size() - total);
							if(written <= 0){
								break;
							}
							total += written;
						} while(total < dump.size());
					}
				}
				throw;
			}
			operation.reset();
		} catch(MySqlException &e){
			LOG_POSEIDON_ERROR("MySqlException thrown in MySQL daemon: code = ", e.code(),
				", what = ", e.what());
			conn.reset();
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown in MySQL daemon: what = ", e.what());
			conn.reset();
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown in MySQL daemon.");
			conn.reset();
		}
	}
exit_loop:

	if(!m_queue.empty()){
		LOG_POSEIDON_WARN("There are still ", m_queue.size(), " object(s) in MySQL queue");
		m_queue.clear();
	}
	m_queueEmpty.notify_all();

	LOG_POSEIDON_INFO("MySQL thread ", m_index, " stopped.");
}

volatile bool g_running = false;
std::vector<boost::shared_ptr<MySqlThread> > g_threads;

void commitOperation(const char *table, boost::shared_ptr<OperationBase> operation, bool urgent){
	if(g_threads.empty()){
		DEBUG_THROW(Exception, SharedNts::observe("No MySQL thread is running"));
	}

	std::size_t threadIndex = 0;
	const char *p = table;
	while(*p){
		threadIndex += (unsigned char)*p;
		++p;
	}
	threadIndex %= g_threads.size();

	LOG_POSEIDON_DEBUG("Assigning MySQL table `", table, "` to thread ", threadIndex);
	g_threads.at(threadIndex)->addOperation(STD_MOVE(operation), urgent);
}

}

void MySqlDaemon::start(){
	if(atomicExchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON_INFO("Starting MySQL daemon...");

	AUTO_REF(conf, MainConfig::getConfigFile());

	conf.get(g_mysqlServerAddr, "mysql_server_addr");
	LOG_POSEIDON_DEBUG("MySQL server addr = ", g_mysqlServerAddr);

	conf.get(g_mysqlServerPort, "mysql_server_port");
	LOG_POSEIDON_DEBUG("MySQL server port = ", g_mysqlServerPort);

	conf.get(g_mysqlUsername, "mysql_username");
	LOG_POSEIDON_DEBUG("MySQL username = ", g_mysqlUsername);

	conf.get(g_mysqlPassword, "mysql_password");
	LOG_POSEIDON_DEBUG("MySQL password = ", g_mysqlPassword);

	conf.get(g_mysqlSchema, "mysql_schema");
	LOG_POSEIDON_DEBUG("MySQL schema = ", g_mysqlSchema);

	conf.get(g_mysqlUseSsl, "mysql_use_ssl");
	LOG_POSEIDON_DEBUG("MySQL use ssl = ", g_mysqlUseSsl);

	conf.get(g_mysqlCharset, "mysql_charset");
	LOG_POSEIDON_DEBUG("MySQL charset = ", g_mysqlCharset);

	conf.get(g_mysqlDumpDir, "mysql_dump_dir");
	LOG_POSEIDON_DEBUG("MySQL dump dir = ", g_mysqlDumpDir);

	conf.get(g_mysqlMaxThreads, "mysql_max_threads");
	LOG_POSEIDON_DEBUG("MySQL max threads = ", g_mysqlMaxThreads);

	conf.get(g_mysqlSaveDelay, "mysql_save_delay");
	LOG_POSEIDON_DEBUG("MySQL save delay = ", g_mysqlSaveDelay);

	conf.get(g_mysqlMaxReconnDelay, "mysql_max_reconn_delay");
	LOG_POSEIDON_DEBUG("MySQL max reconnect delay = ", g_mysqlMaxReconnDelay);

	conf.get(g_mysqlRetryCount, "mysql_retry_count");
	LOG_POSEIDON_DEBUG("MySQL retry count = ", g_mysqlRetryCount);

	char temp[256];
	unsigned len = formatTime(temp, sizeof(temp), getLocalTime(), false);

	std::string dumpPath;
	dumpPath.assign(g_mysqlDumpDir);
	dumpPath.push_back('/');
	dumpPath.append(temp, len);
	dumpPath.append(".log");

	LOG_POSEIDON_INFO("Creating SQL dump file: ", dumpPath);
	if(!g_dumpFile.reset(::open(dumpPath.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_EXCL, 0644))){
		const int errCode = errno;
		LOG_POSEIDON_FATAL("Error creating SQL dump file: errno = ", errCode,
			", description = ", getErrorDesc(errCode));
		std::abort();
	}

	g_threads.resize(std::max<std::size_t>(g_mysqlMaxThreads, 1));
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		LOG_POSEIDON_INFO("Creating MySQL thread ", i);
		g_threads[i] = boost::make_shared<MySqlThread>(i);
	}
}
void MySqlDaemon::stop(){
	if(atomicExchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON_INFO("Stopping MySQL daemon...");

	for(std::size_t i = 0; i < g_threads.size(); ++i){
		g_threads[i]->shutdown();
	}
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		LOG_POSEIDON_INFO("Stopping MySQL thread ", i);
		if(g_threads[i]->joinable()){
			g_threads[i]->join();
		}
	}
	g_threads.clear();
}

std::vector<MySqlSnapshotItem> MySqlDaemon::snapshot(){
	std::vector<MySqlSnapshotItem> ret(g_threads.size());
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		AUTO_REF(item, ret[i]);
		item.index = i;
		item.pendingOperations = g_threads[i]->getPendingOperations();
		item.usIdle = g_threads[i]->getTimeIdle();
		item.usWorking = g_threads[i]->getTimeWorking();
	}
	return ret;
}

void MySqlDaemon::waitForAllAsyncOperations(){
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		g_threads[i]->waitTillIdle();
	}
}

void MySqlDaemon::pendForSaving(boost::shared_ptr<const MySqlObjectBase> object, bool toReplace,
	MySqlAsyncSaveCallback callback, MySqlExceptionCallback except)
{
	const AUTO(tableName, object->getTableName());
	const bool urgent = callback;
	commitOperation(tableName,
		boost::make_shared<SaveOperation>(
			getMonoClock() + g_mysqlSaveDelay * 1000,
			STD_MOVE(object), toReplace, STD_MOVE(callback), STD_MOVE(except)),
		urgent);
}
void MySqlDaemon::pendForLoading(boost::shared_ptr<MySqlObjectBase> object, std::string query,
	MySqlAsyncLoadCallback callback, MySqlExceptionCallback except)
{
	const AUTO(tableName, object->getTableName());
	commitOperation(tableName,
		boost::make_shared<LoadOperation>(
			STD_MOVE(object), STD_MOVE(query), STD_MOVE(callback), STD_MOVE(except)),
		true);
}
void MySqlDaemon::pendForBatchLoading(boost::shared_ptr<MySqlObjectBase> (*factory)(),
	const char *tableHint, std::string query,
	MySqlBatchAsyncLoadCallback callback, MySqlExceptionCallback except)
{
	commitOperation(tableHint,
		boost::make_shared<BatchLoadOperation>(
			factory, STD_MOVE(query), STD_MOVE(callback), STD_MOVE(except)),
		true);
}
