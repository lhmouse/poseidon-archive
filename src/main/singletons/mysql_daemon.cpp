// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "mysql_daemon.hpp"
#include "main_config.hpp"
#include <boost/thread.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/bind.hpp>
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

std::string	g_mysqlDumpDir			= "sql_dump";
std::size_t	g_mysqlMaxThreads		= 3;
std::size_t	g_mysqlSaveDelay		= 5000;
std::size_t	g_mysqlMaxReconnDelay	= 60000;
std::size_t	g_mysqlRetryCount		= 3;

class MySqlThread;
class AssignmentItem;

struct MySqlThreadDecrementer {
	CONSTEXPR MySqlThread *operator()() const NOEXCEPT {
		return NULLPTR;
	}
	void operator()(MySqlThread *thread) const NOEXCEPT;
};

typedef UniqueHandle<MySqlThreadDecrementer> AssignedThread;

struct AssignmentItemDecrementer {
	CONSTEXPR AssignmentItem *operator()() const NOEXCEPT {
		return NULLPTR;
	}
	void operator()(AssignmentItem *assignment) const NOEXCEPT;
};

typedef UniqueHandle<AssignmentItemDecrementer> LockingAssignment;

class OperationBase : NONCOPYABLE {
private:
	AssignedThread m_assignedThread;
	LockingAssignment m_lockingAssignment;

public:
	virtual ~OperationBase(){
	}

public:
	void setAssignedThread(AssignedThread thread) NOEXCEPT {
		m_assignedThread.swap(thread);
	}
	void setLockingAssignment(LockingAssignment lockingAssignment) NOEXCEPT {
		m_lockingAssignment.swap(lockingAssignment);
	}

	virtual bool shouldExecuteNow() const = 0;
	virtual void execute(std::string &query, MySqlConnection &conn) = 0;
};

class SaveCallbackJob : public JobBase {
private:
	MySqlAsyncSaveCallback m_callback;
	bool m_succeeded;
	unsigned long long m_autoIncrementId;

public:
	SaveCallbackJob(Move<MySqlAsyncSaveCallback> callback,
		bool succeeded, unsigned long long autoIncrementId)
		: m_succeeded(succeeded), m_autoIncrementId(autoIncrementId)
	{
		swap(m_callback, callback);
	}

private:
	void perform(){
		PROFILE_ME;

		m_callback(m_succeeded, m_autoIncrementId);
	}
};

class SaveOperation : public OperationBase {
private:
	const boost::uint64_t m_dueTime;
	const boost::shared_ptr<const MySqlObjectBase> m_object;
	const bool m_replaces;
	MySqlAsyncSaveCallback m_callback;

public:
	SaveOperation(boost::shared_ptr<const MySqlObjectBase> object,
		bool replaces, Move<MySqlAsyncSaveCallback> callback)
		: m_dueTime(getMonoClock() + g_mysqlSaveDelay * 1000)
		, m_object(STD_MOVE(object)), m_replaces(replaces)
	{
		swap(m_callback, callback);

		m_object->setContext(this);
	}

private:
	bool shouldExecuteNow() const {
		return m_dueTime <= getMonoClock();
	}
	void execute(std::string &query, MySqlConnection &conn){
		PROFILE_ME;

		if(m_object->getContext() != this){
			// 使用写入合并策略，丢弃当前的写入操作（认为已成功）。
			return;
		}

		m_object->syncGenerateSql(query, m_replaces);
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
			const AUTO(autoId, conn.getInsertId());
			pendJob(boost::make_shared<SaveCallbackJob>(STD_MOVE(m_callback), succeeded, autoId));
		}
	}
};

class LoadCallbackJob : public JobBase {
private:
	MySqlAsyncLoadCallback m_callback;
	bool m_found;

public:
	LoadCallbackJob(Move<MySqlAsyncLoadCallback> callback, bool found)
		: m_found(found)
	{
		swap(m_callback, callback);
	}

private:
	void perform(){
		PROFILE_ME;

		m_callback(m_found);
	}
};

class LoadOperation : public OperationBase {
private:
	const boost::shared_ptr<MySqlObjectBase> m_object;
	const std::string m_query;
	MySqlAsyncLoadCallback m_callback;

public:
	LoadOperation(boost::shared_ptr<MySqlObjectBase> object,
		std::string query, Move<MySqlAsyncLoadCallback> callback)
		: m_object(STD_MOVE(object)), m_query(STD_MOVE(query))
	{
		swap(m_callback, callback);
	}

private:
	bool shouldExecuteNow() const {
		return true;
	}
	void execute(std::string &query, MySqlConnection &conn){
		PROFILE_ME;

		query = m_query;

		LOG_POSEIDON_INFO("MySQL load: table = ", m_object->getTableName(), ", query = ", query);
		conn.executeSql(query);
		const bool found = conn.fetchRow();
		if(found){
			m_object->syncFetch(conn);
			m_object->enableAutoSaving();
		}

		if(m_callback){
			pendJob(boost::make_shared<LoadCallbackJob>(STD_MOVE(m_callback), found));
		}
	}
};

class BatchAsyncLoadCallbackJob : public JobBase {
private:
	MySqlBatchAsyncLoadCallback m_callback;
	std::vector<boost::shared_ptr<MySqlObjectBase> > m_objects;

public:
	BatchAsyncLoadCallbackJob(Move<MySqlBatchAsyncLoadCallback> callback,
		std::vector<boost::shared_ptr<MySqlObjectBase> > objects)
		: m_objects(STD_MOVE(objects))
	{
		swap(m_callback, callback);
	}

private:
	void perform(){
		PROFILE_ME;

		m_callback(STD_MOVE(m_objects));
	}
};

class BatchAsyncLoadOperation : public OperationBase {
private:
	const std::string m_query;
	boost::shared_ptr<MySqlObjectBase> (*const m_factory)();

	MySqlBatchAsyncLoadCallback m_callback;

public:
	BatchAsyncLoadOperation(std::string query, boost::shared_ptr<MySqlObjectBase> (*factory)(),
		Move<MySqlBatchAsyncLoadCallback> callback)
		: m_query(STD_MOVE(query)), m_factory(factory)
	{
		swap(m_callback, callback);
	}

private:
	bool shouldExecuteNow() const {
		return true;
	}
	void execute(std::string &query, MySqlConnection &conn){
		PROFILE_ME;

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
			pendJob(boost::make_shared<BatchAsyncLoadCallbackJob>(
				STD_MOVE(m_callback), STD_MOVE(objects)));
		}
	}
};

class MySqlThread : NONCOPYABLE {
	friend MySqlThreadDecrementer;

public:
	class Profiler : NONCOPYABLE {
	private:
		MySqlThread &m_owner;

	public:
		explicit Profiler(MySqlThread &owner)
			: m_owner(owner)
		{
			m_owner.flushProfile();
			++m_owner.m_workingCount;
		}
		~Profiler(){
			m_owner.flushProfile();
			--m_owner.m_workingCount;
		}
	};

private:
	const unsigned m_id;

	boost::thread m_thread;
	volatile bool m_running;

	mutable boost::mutex m_mutex;
	std::deque<boost::shared_ptr<OperationBase> > m_queue;
	volatile std::size_t m_pendingOperations; // 调度提示。
	mutable volatile bool m_urgentMode; // 无视延迟写入，一次性处理队列中所有操作。
	mutable boost::condition_variable m_newAvail;
	mutable boost::condition_variable m_queueEmpty;

	// 性能统计。
	mutable boost::uint64_t m_timeFlushed;
	std::size_t m_workingCount;
	mutable volatile boost::uint64_t m_timeIdle;
	mutable volatile boost::uint64_t m_timeWorking;

public:
	explicit MySqlThread(unsigned id)
		: m_id(id)
		, m_running(false), m_pendingOperations(0), m_urgentMode(false)
		, m_timeFlushed(getMonoClock()), m_workingCount(0), m_timeIdle(0), m_timeWorking(0)
	{
	}
	~MySqlThread(){
		flushProfile();
	}

private:
	AssignedThread increment(){
		atomicAdd(m_pendingOperations, 1, ATOMIC_RELAXED);
		return AssignedThread(this);
	}
	void decrement() NOEXCEPT {
		atomicSub(m_pendingOperations, 1, ATOMIC_RELAXED);
	}

	void flushProfile() const NOEXCEPT {
		const AUTO(now, getMonoClock());
		const AUTO(delta, now - m_timeFlushed);
		m_timeFlushed = now;
		if(m_workingCount == 0){
			atomicAdd(m_timeIdle, delta, ATOMIC_RELAXED);
		} else {
			atomicAdd(m_timeWorking, delta, ATOMIC_RELAXED);
		}
	}

	void operationLoop();
	void threadProc();

public:
	boost::uint64_t getPendingOperations() const {
		return atomicLoad(m_pendingOperations, ATOMIC_RELAXED);
	}

	boost::uint64_t getTimeIdle() const {
		return atomicLoad(m_timeIdle, ATOMIC_RELAXED);
	}
	boost::uint64_t getTimeWorking() const {
		return atomicLoad(m_timeWorking, ATOMIC_RELAXED);
	}

	void start(){
		if(atomicExchange(m_running, true, ATOMIC_ACQ_REL) != false){
			return;
		}
		boost::thread(boost::bind(&MySqlThread::threadProc, this)).swap(m_thread);
	}
	void stop(){
		if(atomicExchange(m_running, false, ATOMIC_ACQ_REL) == false){
			return;
		}
		{
			const boost::mutex::scoped_lock lock(m_mutex);
			atomicStore(m_urgentMode, true, ATOMIC_RELEASE);
			m_newAvail.notify_all();
		}
	}
	void join(){
		waitTillIdle();
		if(m_thread.joinable()){
			m_thread.join();
		}
		boost::thread().swap(m_thread);
	}

	void addOperation(boost::shared_ptr<OperationBase> operation, bool urgent){
		const boost::mutex::scoped_lock lock(m_mutex);
		m_queue.push_back(operation);
		operation->setAssignedThread(increment());
		if(urgent){
			atomicStore(m_urgentMode, true, ATOMIC_RELEASE);
		}
		m_newAvail.notify_all();
	}

	void waitTillIdle() const {
		atomicStore(m_urgentMode, true, ATOMIC_RELEASE);
		boost::mutex::scoped_lock lock(m_mutex);
		while(getPendingOperations() != 0){
			atomicStore(m_urgentMode, true, ATOMIC_RELEASE);
			m_newAvail.notify_all();
			m_queueEmpty.wait(lock);
		}
	}
};

boost::mutex g_dumpMutex;
UniqueFile g_dumpFile;

volatile bool g_running = false;
std::vector<boost::shared_ptr<MySqlThread> > g_threads;

// 根据表名分给不同的线程。
class AssignmentItem : NONCOPYABLE
	, public boost::enable_shared_from_this<AssignmentItem>
{
	friend AssignmentItemDecrementer;

private:
	const char *const m_table;

	mutable boost::mutex m_mutex;
	std::size_t m_pendingOperations;
	boost::shared_ptr<MySqlThread> m_thread;

public:
	explicit AssignmentItem(const char *table)
		: m_table(table), m_pendingOperations(0)
	{
	}

private:
	LockingAssignment increment(){
		const boost::mutex::scoped_lock lock(m_mutex);
		if(m_pendingOperations == 0){
			if(g_threads.empty()){
				DEBUG_THROW(Exception, "No threads available");
			}
			// 指派给最空闲的线程。
			std::size_t picked = 0;
			AUTO(minPendingOperations, g_threads.front()->getPendingOperations());
			LOG_POSEIDON_TRACE("MySQL thread 0 pending operations: ", minPendingOperations);
			for(std::size_t i = 1; i < g_threads.size(); ++i){
				const AUTO(myPendingOperations, g_threads[i]->getPendingOperations());
				LOG_POSEIDON_TRACE("MySQL thread ", i, " pending operations: ", myPendingOperations);
				if(minPendingOperations > myPendingOperations){
					picked = i;
					minPendingOperations = myPendingOperations;
				}
			}
			m_thread = g_threads[picked];
			LOG_POSEIDON_DEBUG("Assigned table `", m_table, "` to thread ", picked);
		}
		++m_pendingOperations;
		return LockingAssignment(this);
	};
	void decrement() NOEXCEPT {
		const boost::mutex::scoped_lock lock(m_mutex);
		--m_pendingOperations;
		if(m_pendingOperations == 0){
			m_thread.reset();
			LOG_POSEIDON_DEBUG("No more pending operations: ", m_table);
		}
	}

public:
	const char *getTable() const {
		return m_table;
	}

	void commit(boost::shared_ptr<OperationBase> operation, bool urgent){
		operation->setLockingAssignment(increment());
		m_thread->addOperation(STD_MOVE(operation), urgent);
	}
};

void MySqlThreadDecrementer::operator()(MySqlThread *thread) const NOEXCEPT {
	thread->decrement();
}

void AssignmentItemDecrementer::operator()(AssignmentItem *assignment) const NOEXCEPT {
	assignment->decrement();
}

boost::mutex g_assignmentMutex;
std::vector<boost::shared_ptr<AssignmentItem> > g_assignments;

AssignmentItem &getAssignmentForTable(const char *table){
	const boost::mutex::scoped_lock lock(g_assignmentMutex);
	AUTO(lower, g_assignments.begin());
	AUTO(upper, g_assignments.end());
	for(;;){
		if(lower == upper){
			lower = g_assignments.insert(lower, boost::make_shared<AssignmentItem>(table));
			break;
		}
		const AUTO(middle, lower + (upper - lower) / 2);
		const int result = std::strcmp(table, (*middle)->getTable());
		if(result == 0){
			lower = middle;
			break;
		} else if(result < 0){
			upper = middle;
		} else {
			lower = middle + 1;
		}
	}
	return **lower;
}

void MySqlThread::operationLoop(){
	MySqlThreadContext context;

	boost::scoped_ptr<MySqlConnection> conn;
	std::size_t reconnectDelay = 0;
	bool discardConnection = false;

	std::deque<boost::shared_ptr<OperationBase> > queue;
	std::size_t retryCount = 0;
	for(;;){
		try {
			flushProfile();

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

			if(queue.empty()){
				boost::mutex::scoped_lock lock(m_mutex);
				if(m_queue.empty()){
					m_queueEmpty.notify_all();
				}
				while(m_queue.empty()){
					if(!atomicLoad(m_running, ATOMIC_ACQUIRE)){
						goto exit_loop;
					}
					atomicStore(m_urgentMode, false, ATOMIC_RELEASE);
					flushProfile();
					m_newAvail.timed_wait(lock, boost::posix_time::seconds(1));
				}
				queue.swap(m_queue);
			}

			if(!queue.front()->shouldExecuteNow() && !atomicLoad(m_urgentMode, ATOMIC_ACQUIRE)){
				boost::mutex::scoped_lock lock(m_mutex);
				m_newAvail.timed_wait(lock, boost::posix_time::seconds(1));
				continue;
			}

			unsigned mysqlErrCode = 99999;
			std::string mysqlErrMsg;
			std::string query;
			try {
				try {
					const Profiler profiler(*this);
					queue.front()->execute(query, *conn);
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

					queue.pop_front();

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
			queue.pop_front();
		} catch(MySqlException &e){
			LOG_POSEIDON_ERROR("MySqlException thrown in MySQL daemon: code = ", e.code(),
				", what = ", e.what());
			discardConnection = true;
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown in MySQL daemon: what = ", e.what());
			discardConnection = true;
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown in MySQL daemon.");
			discardConnection = true;
		}

		if(discardConnection){
			discardConnection = false;
			if(conn){
				LOG_POSEIDON_INFO("The connection was left in an indeterminate state. Discard it.");
				conn.reset();
			}
		}
	}
exit_loop:

	if(!m_queue.empty()){
		LOG_POSEIDON_ERROR("There are still ", m_queue.size(), " object(s) in MySQL queue");
		m_queue.clear();
	}
	m_queueEmpty.notify_all();
}

void MySqlThread::threadProc(){
	PROFILE_ME;
	Logger::setThreadTag(" D  "); // Database
	LOG_POSEIDON_INFO("MySQL thread ", m_id, " started.");

	operationLoop();

	LOG_POSEIDON_INFO("MySQL thread ", m_id, " stopped.");
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
		boost::make_shared<MySqlThread>(i).swap(g_threads[i]);
		g_threads[i]->start();

		LOG_POSEIDON_INFO("Created MySQL thread ", i);
	}
}
void MySqlDaemon::stop(){
	if(atomicExchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON_INFO("Stopping MySQL daemon...");

	for(std::size_t i = 0; i < g_threads.size(); ++i){
		g_threads[i]->stop();
	}
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		g_threads[i]->join();

		LOG_POSEIDON_INFO("Shut down MySQL thread ", i);
	}
	g_threads.clear();
}

std::vector<MySqlSnapshotItem> MySqlDaemon::snapshot(){
	std::vector<MySqlSnapshotItem> ret;
	ret.reserve(g_threads.size());
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		ret.push_back(MySqlSnapshotItem());
		AUTO_REF(item, ret.back());
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

void MySqlDaemon::pendForSaving(boost::shared_ptr<const MySqlObjectBase> object, bool replaces,
	MySqlAsyncSaveCallback callback)
{
	AUTO_REF(assignment, getAssignmentForTable(object->getTableName()));
	const bool urgent = !!callback;
	assignment.commit(boost::make_shared<SaveOperation>(
		STD_MOVE(object), replaces, STD_MOVE(callback)), urgent);
}
void MySqlDaemon::pendForLoading(boost::shared_ptr<MySqlObjectBase> object, std::string query,
	MySqlAsyncLoadCallback callback)
{
	AUTO_REF(assignment, getAssignmentForTable(object->getTableName()));
	assignment.commit(boost::make_shared<LoadOperation>(
		STD_MOVE(object), STD_MOVE(query), STD_MOVE(callback)), true);
}
void MySqlDaemon::pendForBatchAsyncLoading(const char *tableHint, std::string query,
	boost::shared_ptr<MySqlObjectBase> (*factory)(), MySqlBatchAsyncLoadCallback callback)
{
	AUTO_REF(assignment, getAssignmentForTable(tableHint));
	assignment.commit(boost::make_shared<BatchAsyncLoadOperation>(
		STD_MOVE(query), factory, STD_MOVE(callback)), true);
}
