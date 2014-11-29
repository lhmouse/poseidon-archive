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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "main_config.hpp"
#include "../mysql/object_base.hpp"
#define POSEIDON_MYSQL_OBJECT_IMPL_
#include "../mysql/object_impl.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../utilities.hpp"
using namespace Poseidon;

namespace {

std::string g_mySqlServer			= "tcp://localhost:3306";
std::string g_mySqlUsername			= "root";
std::string g_mySqlPassword			= "root";
std::string g_mySqlSchema			= "poseidon";

std::string g_mySqlDumpDir			= "sql_dump";
std::size_t g_mySqlMaxThreads		= 3;
std::size_t g_mySqlSaveDelay		= 5000;
std::size_t g_mySqlMaxReconnDelay	= 60000;
std::size_t g_mySqlRetryCount		= 3;

class OperationBase : boost::noncopyable {
public:
	virtual ~OperationBase(){
	}

public:
	virtual bool shouldExecuteNow() const = 0;
	virtual void execute(std::string &sqlStatement, sql::Connection *connection) = 0;
};

class LoadOperation
	// 作为两步使用，塞到了同一个类里面。基类是按照这两个用途的先后顺序排列的，不要弄混。
	: public OperationBase, public JobBase
{
private:
	boost::shared_ptr<MySqlObjectBase> m_object;
	std::string m_filter;
	MySqlAsyncLoadCallback m_callback;

public:
	LoadOperation(boost::shared_ptr<MySqlObjectBase> object,
		std::string filter, Move<MySqlAsyncLoadCallback> callback)
		: m_object(STD_MOVE(object)), m_filter(STD_MOVE(filter))
	{
		callback.swap(m_callback);
	}

public:
	bool shouldExecuteNow() const {
		return true;
	}
	void execute(std::string &sqlStatement, sql::Connection *connection){
		PROFILE_ME;

		LOG_POSEIDON_INFO("MySQL load: table = ", m_object->getTableName(), ", filter = ", m_filter);
		m_object->syncLoad(sqlStatement, connection, m_filter.c_str());
		m_object->enableAutoSaving();

		JobBase::pend();
	}

	void perform(){
		PROFILE_ME;

		m_callback(STD_MOVE(m_object));
	}
};

class SaveOperation : public OperationBase {
private:
	const boost::uint64_t m_dueTime;
	const boost::shared_ptr<const MySqlObjectBase> m_object;

public:
	explicit SaveOperation(boost::shared_ptr<const MySqlObjectBase> object)
		: m_dueTime(getMonoClock() + g_mySqlSaveDelay * 1000), m_object(STD_MOVE(object))
	{
		MySqlObjectImpl::setContext(*m_object, this);
	}

public:
	bool shouldExecuteNow() const {
		return m_dueTime <= getMonoClock();
	}
	void execute(std::string &sqlStatement, sql::Connection *connection){
		PROFILE_ME;

		if(MySqlObjectImpl::getContext(*m_object) != this){
			// 使用写入合并策略，丢弃当前的写入操作（认为已成功）。
			return;
		}
		LOG_POSEIDON_INFO("MySQL save: table = ", m_object->getTableName());
		m_object->syncSave(sqlStatement, connection);
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
ScopedFile g_dumpFile;

volatile bool g_running = false;
std::vector<boost::shared_ptr<MySqlThread> > g_threads;

// 根据表名分给不同的线程。
boost::mutex g_assignmentMutex;
std::vector<std::pair<const char *, boost::shared_ptr<MySqlThread> > > g_assignments;

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

void MySqlThread::operationLoop(){
	boost::scoped_ptr<sql::Connection> connection;
	std::size_t reconnectDelay = 0;
	bool discardConnection = false;

	if(!getMySqlConnection(connection)){
		LOG_POSEIDON_FATAL("Failed to connect MySQL server. Bailing out.");
		std::abort();
	} else {
		const boost::mutex::scoped_lock lock(m_mutex);
		m_queueEmpty.notify_all();
	}

	std::deque<boost::shared_ptr<OperationBase> > queue;
	std::size_t retryCount = 0;

	for(;;){
		try {
			if(!connection){
				LOG_POSEIDON_WARN("Lost connection to MySQL server. Reconnecting...");

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
				if(!getMySqlConnection(connection)){
					if(!atomicLoad(m_running)){
						LOG_POSEIDON_WARN("Shutting down...");
						break;
					}
					continue;
				}
				reconnectDelay = 0;
			}

			if(queue.empty()){
				boost::mutex::scoped_lock lock(m_mutex);
				if(m_queue.empty()){
					atomicStore(m_urgentMode, false);
					do {
						if(!atomicLoad(m_running)){
							goto exit_loop;
						}
						m_newAvail.timed_wait(lock, boost::posix_time::seconds(1));
					} while(m_queue.empty());
				}
				queue.swap(m_queue);
				m_queueEmpty.notify_all();
			}

			if(!queue.front()->shouldExecuteNow() && !atomicLoad(m_urgentMode)){
				boost::mutex::scoped_lock lock(m_mutex);
				m_newAvail.timed_wait(lock, boost::posix_time::seconds(1));
				continue;
			}

			unsigned mySqlErrno = 99999;
			std::string mySqlError;
			std::string sqlStatement;
			try {
				try {
					const Stopwatch watch(*this);
					queue.front()->execute(sqlStatement, connection.get());
				} catch(sql::SQLException &e){
					mySqlErrno = e.getErrorCode();
					mySqlError = e.what();
					throw;
				} catch(std::exception &e){
					mySqlError = e.what();
					throw;
				} catch(...){
					mySqlError = "Unknown exception";
					throw;
				}
			} catch(...){
				bool retry = true;
				if(retryCount == 0){
					if(g_mySqlRetryCount != 0){
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
					unsigned len = std::sprintf(temp, "%05u", mySqlErrno);
					std::string dump;
					dump.reserve(1024);
					dump.assign("-- Error code = ");
					dump.append(temp, len);
					dump.append(", Description = ");
					dump.append(mySqlError);
					dump.append("\n");
					dump.append(sqlStatement);
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
		} catch(sql::SQLException &e){
			LOG_POSEIDON_ERROR("SQLException thrown in MySQL daemon: code = ", e.getErrorCode(),
				", state = ", e.getSQLState(), ", what = ", e.what());
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
			if(connection){
				LOG_POSEIDON_INFO("The connection was left in an indeterminate state. Discard it.");
				connection.reset();
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

	::get_driver_instance()->threadInit();
	operationLoop();
	::get_driver_instance()->threadEnd();

	LOG_POSEIDON_INFO("MySQL daemon stopped.");
}

boost::shared_ptr<MySqlThread> pickThreadForTable(const char *table){
	const boost::mutex::scoped_lock lock(g_assignmentMutex);
	if(g_threads.empty()){
		DEBUG_THROW(Exception, "No threads available");
	}
	AUTO(lower, g_assignments.begin());
	AUTO(upper, g_assignments.end());
	boost::shared_ptr<MySqlThread> thread;
	for(;;){
		if(lower == upper){
			thread = g_threads.front();
			for(AUTO(it, g_threads.begin() + 1); it != g_threads.end(); ++it){
				if((*it)->getBusyTime() < thread->getBusyTime()){
					thread = *it;
				}
			}
			g_assignments.insert(lower, std::make_pair(table, thread));
			break;
		}
		const AUTO(middle, lower + (upper - lower) / 2);
		const int result = std::strcmp(table, middle->first);
		if(result == 0){
			thread = middle->second;
			LOG_POSEIDON_DEBUG("MySQL table `", table, "` has bee assigned to thread ",
				middle - g_assignments.begin());
			break;
		} else if(result < 0){
			upper = middle;
		} else {
			lower = middle + 1;
		}
	}
	return thread;
}

}

void MySqlDaemon::start(){
	if(atomicExchange(g_running, true) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON_INFO("Starting MySQL daemon...");

	// 全局初始化。
	::get_driver_instance()->threadInit();

	AUTO_REF(conf, MainConfig::getConfigFile());

	conf.get(g_mySqlServer, "mysql_server");
	LOG_POSEIDON_DEBUG("MySQL server = ", g_mySqlServer);

	conf.get(g_mySqlUsername, "mysql_username");
	LOG_POSEIDON_DEBUG("MySQL username = ", g_mySqlUsername);

	conf.get(g_mySqlPassword, "mysql_password");
	LOG_POSEIDON_DEBUG("MySQL password = ", g_mySqlPassword);

	conf.get(g_mySqlSchema, "mysql_schema");
	LOG_POSEIDON_DEBUG("MySQL schema = ", g_mySqlSchema);

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

		LOG_POSEIDON_INFO("Shut down MySQL thread ", i);
	}
	g_threads.clear();

	::get_driver_instance()->threadEnd();
}

void MySqlDaemon::waitForAllAsyncOperations(){
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		g_threads[i]->waitTillIdle();
	}
}

void MySqlDaemon::pendForSaving(boost::shared_ptr<const MySqlObjectBase> object){
	const AUTO(table, object->getTableName());
	pickThreadForTable(table)->addOperation(
		boost::make_shared<SaveOperation>(STD_MOVE(object)), false);
}
void MySqlDaemon::pendForLoading(boost::shared_ptr<MySqlObjectBase> object,
	std::string filter, MySqlAsyncLoadCallback callback)
{
	const AUTO(table, object->getTableName());
	pickThreadForTable(table)->addOperation(
		boost::make_shared<LoadOperation>(STD_MOVE(object), STD_MOVE(filter), STD_MOVE(callback)), true);
}
