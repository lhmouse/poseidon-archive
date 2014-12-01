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

std::string	g_mySqlServerAddr		= "localhost";
unsigned	g_mySqlServerPort		= 3306;
std::string	g_mySqlUsername			= "root";
std::string	g_mySqlPassword			= "root";
std::string	g_mySqlSchema			= "poseidon";
bool		g_mySqlUseSsl			= false;
std::string	g_mySqlCharset			= "utf8";

std::string	g_mySqlDumpDir			= "sql_dump";
std::size_t	g_mySqlMaxThreads		= 3;
std::size_t	g_mySqlSaveDelay		= 5000;
std::size_t	g_mySqlMaxReconnDelay	= 60000;
std::size_t	g_mySqlRetryCount		= 3;

class AssignmentItem;

struct AssignmentDecrementer {
	CONSTEXPR AssignmentItem *operator()() const NOEXCEPT {
		return NULLPTR;
	}
	void operator()(AssignmentItem *assignment) const NOEXCEPT;
};

typedef UniqueHandle<AssignmentDecrementer> AssignmentLock;

class OperationBase : boost::noncopyable {
private:
	AssignmentLock m_assignmentLock;

public:
	virtual ~OperationBase(){
	}

public:
	void setAssignmentLock(AssignmentLock assignmentLock) NOEXCEPT {
		swap(m_assignmentLock, assignmentLock);
	}

	virtual bool shouldExecuteNow() const = 0;
	virtual void execute(std::string &query, MySqlConnection &conn) = 0;
};

class SaveOperation : public OperationBase {
private:
	const boost::uint64_t m_dueTime;
	const boost::shared_ptr<const MySqlObjectBase> m_object;

public:
	explicit SaveOperation(boost::shared_ptr<const MySqlObjectBase> object)
		: m_dueTime(getMonoClock() + g_mySqlSaveDelay * 1000), m_object(STD_MOVE(object))
	{
		m_object->setContext(this);
	}

public:
	bool shouldExecuteNow() const {
		return m_dueTime <= getMonoClock();
	}
	void execute(std::string &query, MySqlConnection &conn){
		PROFILE_ME;

		if(m_object->getContext() != this){
			// 使用写入合并策略，丢弃当前的写入操作（认为已成功）。
			return;
		}
		LOG_POSEIDON_INFO("MySQL save: table = ", m_object->getTableName());
		m_object->syncSave(query, conn);
	}
};

class LoadOperation
	// 作为两步使用，塞到了同一个类里面。基类是按照这两个用途的先后顺序排列的，不要弄混。
	: public OperationBase, public JobBase
{
private:
	boost::shared_ptr<MySqlObjectBase> m_object;
	std::string m_query;

	MySqlAsyncLoadCallback m_callback;
	bool m_result;

public:
	LoadOperation(boost::shared_ptr<MySqlObjectBase> object,
		std::string query, Move<MySqlAsyncLoadCallback> callback)
		: m_object(STD_MOVE(object)), m_query(STD_MOVE(query))
	{
		swap(m_callback, callback);
	}

public:
	bool shouldExecuteNow() const {
		return true;
	}
	void execute(std::string &query, MySqlConnection &conn){
		PROFILE_ME;

		query.swap(m_query);
		LOG_POSEIDON_INFO("MySQL load: table = ", m_object->getTableName(), ", query = ", query);
		conn.executeSql(query);
		conn.waitForResult();
		m_result = conn.fetchRow();
		if(m_result){
			m_object->syncFetch(conn);
			m_object->enableAutoSaving();
		}

		JobBase::pend();
	}

	void perform(){
		PROFILE_ME;

		m_callback(STD_MOVE(m_object), m_result);
	}
};

class BatchLoadOperation
	: public OperationBase, public JobBase
{
private:
	std::string m_query;
	MySqlObjectFactoryProc m_factory;

	std::vector<boost::shared_ptr<MySqlObjectBase> > m_objects;
	MySqlBatchAsyncLoadCallback m_callback;

public:
	BatchLoadOperation(std::string query, MySqlObjectFactoryProc factory,
		Move<MySqlBatchAsyncLoadCallback> callback)
		: m_query(STD_MOVE(query)), m_factory(factory)
	{
		callback.swap(m_callback);
	}

public:
	bool shouldExecuteNow() const {
		return true;
	}
	void execute(std::string &query, MySqlConnection &conn){
		PROFILE_ME;

		query = m_query;

		LOG_POSEIDON_INFO("MySQL batch load: query = ", m_query);
		conn.executeSql(m_query);
		conn.waitForResult();
		while(conn.fetchRow()){
			AUTO(object, (*m_factory)());
			object->syncFetch(conn);
			object->enableAutoSaving();
			m_objects.push_back(STD_MOVE(object));
		}

		JobBase::pend();
	}

	void perform(){
		PROFILE_ME;

		m_callback(STD_MOVE(m_objects));
	}
};

class MySqlThread : boost::noncopyable {
public:
	class Stopwatch : boost::noncopyable {
	private:
		MySqlThread &m_owner;
		const boost::uint64_t m_begin;

	public:
		explicit Stopwatch(MySqlThread &owner)
			: m_owner(owner), m_begin(getMonoClock())
		{
		}
		~Stopwatch(){
			const boost::uint64_t delta = getMonoClock() - m_begin;
			atomicAdd(m_owner.m_busyTime, delta);

			LOG_POSEIDON_TRACE("MySQL operation executed in ", delta, " us.");
		}
	};

private:
	boost::thread m_thread;
	volatile bool m_running;

	mutable boost::mutex m_mutex;
	std::deque<boost::shared_ptr<OperationBase> > m_queue;
	volatile boost::uint64_t m_busyTime; // 调度提示。
	mutable volatile bool m_urgentMode; // 无视写入合并策略，一次性处理队列中所有操作。
	mutable boost::condition_variable m_newAvail;
	mutable boost::condition_variable m_queueEmpty;

public:
	MySqlThread()
		: m_running(false), m_busyTime(0), m_urgentMode(false)
	{
	}

private:
	void operationLoop();
	void threadProc();

public:
	boost::uint64_t getBusyTime() const {
		return atomicLoad(m_busyTime);
	}

	void start(){
		if(atomicExchange(m_running, true) != false){
			return;
		}
		boost::thread(boost::bind(&MySqlThread::threadProc, this)).swap(m_thread);
	}
	void stop(){
		if(atomicExchange(m_running, false) == false){
			return;
		}
		{
			const boost::mutex::scoped_lock lock(m_mutex);
			atomicStore(m_urgentMode, true);
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
		m_queue.push_back(VAL_INIT);
		m_queue.back().swap(operation);
		if(urgent){
			atomicStore(m_urgentMode, true);
		}
		atomicAdd(m_busyTime, 1000000); // 1000 毫秒轮转。
		m_newAvail.notify_all();
	}

	void waitTillIdle() const {
		atomicStore(m_urgentMode, true);
		boost::mutex::scoped_lock lock(m_mutex);
		while(!m_queue.empty()){
			atomicStore(m_urgentMode, true);
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
class AssignmentItem : boost::noncopyable
	, public boost::enable_shared_from_this<AssignmentItem>
{
	friend AssignmentDecrementer;

private:
	const char *const m_table;

	mutable boost::mutex m_mutex;
	std::size_t m_count;
	boost::shared_ptr<MySqlThread> m_thread;

public:
	explicit AssignmentItem(const char *table)
		: m_table(table), m_count(0)
	{
	}

private:
	AssignmentLock increment(){
		const boost::mutex::scoped_lock lock(m_mutex);
		if(m_count == 0){
			if(g_threads.empty()){
				DEBUG_THROW(Exception, "No threads available");
			}
			// 指派给最空闲的线程。
			std::size_t picked = 0;
			AUTO(minBusyTime, g_threads.front()->getBusyTime());
			LOG_POSEIDON_TRACE("Busy time of MySQL thread 0: ", minBusyTime);
			for(std::size_t i = 1; i < g_threads.size(); ++i){
				const AUTO(myBusyTime, g_threads[i]->getBusyTime());
				LOG_POSEIDON_TRACE("Busy time of MySQL thread ", i, ": ", myBusyTime);
				// 即使是无符号 64 位整数发生了进位，这个结果也是对的。
				// 参考 TCP 中对于报文流水号的处理。
				if(static_cast<boost::int64_t>(minBusyTime - myBusyTime) > 0){
					picked = i;
					minBusyTime = myBusyTime;
				}
			}
			m_thread = g_threads[picked];
			LOG_POSEIDON_DEBUG("Assigned table `", m_table, "` to thread ", picked);
		}
		++m_count;
		return AssignmentLock(this);
	};
	void decrement(){
		const boost::mutex::scoped_lock lock(m_mutex);
		--m_count;
		if(m_count == 0){
			m_thread.reset();
			LOG_POSEIDON_DEBUG("No more pending operations: ", m_table);
		}
	}

public:
	const char *getTable() const {
		return m_table;
	}

	void commit(boost::shared_ptr<OperationBase> operation, bool urgent){
		// 锁定 m_thread。
		AUTO(lock, increment());
		operation->setAssignmentLock(STD_MOVE(lock));

		// 现在访问 m_thread 是安全的。
		m_thread->addOperation(STD_MOVE(operation), urgent);
	}
};

void AssignmentDecrementer::operator()(AssignmentItem *assignment) const NOEXCEPT {
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
			if(!conn){
				LOG_POSEIDON_INFO("Connecting to MySQL server...");

				if(reconnectDelay == 0){
					reconnectDelay = 1;
				} else {
					LOG_POSEIDON_INFO("Will retry after ", reconnectDelay, " milliseconds.");

					boost::mutex::scoped_lock lock(m_mutex);
					m_newAvail.timed_wait(lock, boost::posix_time::milliseconds(reconnectDelay));

					reconnectDelay <<= 1;
					if(reconnectDelay > g_mySqlMaxReconnDelay){
						reconnectDelay = g_mySqlMaxReconnDelay;
					}
				}
				try {
					MySqlConnection::create(conn, context, g_mySqlServerAddr, g_mySqlServerPort,
						g_mySqlUsername, g_mySqlPassword, g_mySqlSchema, g_mySqlUseSsl, g_mySqlCharset);
				} catch(...){
					if(!atomicLoad(m_running)){
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
				while(m_queue.empty()){
					if(!atomicLoad(m_running)){
						goto exit_loop;
					}
					atomicStore(m_urgentMode, false);
					m_newAvail.timed_wait(lock, boost::posix_time::seconds(1));
				}
				queue.swap(m_queue);
				m_queueEmpty.notify_all();
			}

			if(!queue.front()->shouldExecuteNow() && !atomicLoad(m_urgentMode)){
				boost::mutex::scoped_lock lock(m_mutex);
				m_newAvail.timed_wait(lock, boost::posix_time::seconds(1));
				continue;
			}

			unsigned mySqlErrCode = 99999;
			std::string mySqlErrMsg;
			std::string query;
			try {
				try {
					const Stopwatch watch(*this);
					queue.front()->execute(query, *conn);
				} catch(MySqlException &e){
					mySqlErrCode = e.code();
					mySqlErrMsg = e.what();
					throw;
				} catch(std::exception &e){
					mySqlErrMsg = e.what();
					throw;
				} catch(...){
					mySqlErrMsg = "Unknown exception";
					throw;
				}
			} catch(...){
				bool retry = true;
				if(retryCount == 0){
					if(g_mySqlRetryCount == 0){
						retry = false;
					} else {
						retryCount = g_mySqlRetryCount;
					}
				} else {
					if(--retryCount == 0){
						retry = false;
					}
				}
				if(!retry){
					LOG_POSEIDON_WARN("Retry count drops to zero. Give up.");
					queue.pop_front();

					char temp[32];
					unsigned len = std::sprintf(temp, "%05u", mySqlErrCode);
					std::string dump;
					dump.reserve(1024);
					dump.assign("-- Error code = ");
					dump.append(temp, len);
					dump.append(", Description = ");
					dump.append(mySqlErrMsg);
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
}

void MySqlThread::threadProc(){
	PROFILE_ME;
	Logger::setThreadTag(" D  "); // Database
	LOG_POSEIDON_INFO("MySQL daemon started.");

	operationLoop();

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

	conf.get(g_mySqlServerAddr, "mysql_server_addr");
	LOG_POSEIDON_DEBUG("MySQL server addr = ", g_mySqlServerAddr);

	conf.get(g_mySqlServerPort, "mysql_server_port");
	LOG_POSEIDON_DEBUG("MySQL server port = ", g_mySqlServerPort);

	conf.get(g_mySqlUsername, "mysql_username");
	LOG_POSEIDON_DEBUG("MySQL username = ", g_mySqlUsername);

	conf.get(g_mySqlPassword, "mysql_password");
	LOG_POSEIDON_DEBUG("MySQL password = ", g_mySqlPassword);

	conf.get(g_mySqlSchema, "mysql_schema");
	LOG_POSEIDON_DEBUG("MySQL schema = ", g_mySqlSchema);

	conf.get(g_mySqlUseSsl, "mysql_use_ssl");
	LOG_POSEIDON_DEBUG("MySQL use ssl = ", g_mySqlUseSsl);

	conf.get(g_mySqlCharset, "mysql_charset");
	LOG_POSEIDON_DEBUG("MySQL charset = ", g_mySqlCharset);

	conf.get(g_mySqlDumpDir, "mysql_dump_dir");
	LOG_POSEIDON_DEBUG("MySQL dump dir = ", g_mySqlDumpDir);

	conf.get(g_mySqlMaxThreads, "mysql_max_threads");
	LOG_POSEIDON_DEBUG("MySQL max threads = ", g_mySqlMaxThreads);

	conf.get(g_mySqlSaveDelay, "mysql_save_delay");
	LOG_POSEIDON_DEBUG("MySQL save delay = ", g_mySqlSaveDelay);

	conf.get(g_mySqlMaxReconnDelay, "mysql_max_reconn_delay");
	LOG_POSEIDON_DEBUG("MySQL max reconnect delay = ", g_mySqlMaxReconnDelay);

	conf.get(g_mySqlRetryCount, "mysql_retry_count");
	LOG_POSEIDON_DEBUG("MySQL retry count = ", g_mySqlRetryCount);

	char temp[256];
	unsigned len = formatTime(temp, sizeof(temp), getLocalTime(), false);

	std::string dumpPath;
	dumpPath.assign(g_mySqlDumpDir);
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

	g_threads.resize(std::max<std::size_t>(g_mySqlMaxThreads, 1));
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		boost::make_shared<MySqlThread>().swap(g_threads[i]);
		g_threads[i]->start();

		LOG_POSEIDON_INFO("Created MySQL thread ", i);
	}
}
void MySqlDaemon::stop(){
	if(atomicExchange(g_running, false) == false){
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

void MySqlDaemon::waitForAllAsyncOperations(){
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		g_threads[i]->waitTillIdle();
	}
}

void MySqlDaemon::pendForSaving(boost::shared_ptr<const MySqlObjectBase> object){
	AUTO_REF(assignment, getAssignmentForTable(object->getTableName()));
	assignment.commit(boost::make_shared<SaveOperation>(
		STD_MOVE(object)), false);
}
void MySqlDaemon::pendForLoading(boost::shared_ptr<MySqlObjectBase> object,
	std::string query, MySqlAsyncLoadCallback callback)
{
	AUTO_REF(assignment, getAssignmentForTable(object->getTableName()));
	assignment.commit(boost::make_shared<LoadOperation>(
		STD_MOVE(object), STD_MOVE(query), STD_MOVE(callback)), true);
}
void MySqlDaemon::pendForBatchLoading(const char *tableHint,
	std::string query, MySqlObjectFactoryProc factory, MySqlBatchAsyncLoadCallback callback)
{
	AUTO_REF(assignment, getAssignmentForTable(tableHint));
	assignment.commit(boost::make_shared<BatchLoadOperation>(
		STD_MOVE(query), factory, STD_MOVE(callback)), true);
}
