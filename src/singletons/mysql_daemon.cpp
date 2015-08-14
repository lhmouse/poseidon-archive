// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "mysql_daemon.hpp"
#include "main_config.hpp"
#include <boost/container/vector.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mysql/mysqld_error.h>
#include <mysql/errmsg.h>
#include "../mysql/object_base.hpp"
#include "../mysql/exception.hpp"
#include "../mysql/connection.hpp"
#include "../mysql/thread_context.hpp"
#include "../thread.hpp"
#include "../mutex.hpp"
#include "../condition_variable.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../time.hpp"
#include "../errno.hpp"

namespace Poseidon {

namespace {
	std::string		g_serverAddr		= "localhost";
	unsigned		g_serverPort		= 3306;
	std::string		g_username			= "root";
	std::string		g_password			= "root";
	std::string		g_schema			= "poseidon";
	bool			g_useSsl			= false;
	std::string		g_charset			= "utf8";

	std::string		g_dumpDir			= "";
	std::size_t		g_maxThreads		= 3;
	boost::uint64_t	g_saveDelay			= 5000;
	boost::uint64_t	g_reconnDelay		= 10000;
	std::size_t		g_maxRetryCount		= 3;
	boost::uint64_t	g_retryInitDelay	= 1000;

	// 对于日志文件的写操作应当互斥。
	Mutex g_dumpMutex;

	// 数据库线程操作。
	class OperationBase : NONCOPYABLE {
	private:
		const boost::shared_ptr<MySql::Promise> m_promise;

	public:
		explicit OperationBase(boost::shared_ptr<MySql::Promise> promise)
			: m_promise(STD_MOVE(promise))
		{
		}
		virtual ~OperationBase(){
		}

	public:
		const boost::shared_ptr<MySql::Promise> &getPromise() const {
			return m_promise;
		}

		virtual boost::shared_ptr<const MySql::ObjectBase> getCombinableObject() const = 0;
		virtual const char *getTableName() const = 0;
		virtual void execute(std::string &query, const boost::shared_ptr<MySql::Connection> &conn) const = 0;
	};

	class SaveOperation : public OperationBase {
	private:
		const boost::shared_ptr<const MySql::ObjectBase> m_object;
		const bool m_toReplace;

	public:
		SaveOperation(boost::shared_ptr<MySql::Promise> promise,
			boost::shared_ptr<const MySql::ObjectBase> object, bool toReplace)
			: OperationBase(STD_MOVE(promise))
			, m_object(STD_MOVE(object)), m_toReplace(toReplace)
		{
		}

	protected:
		boost::shared_ptr<const MySql::ObjectBase> getCombinableObject() const OVERRIDE {
			return m_object;
		}
		const char *getTableName() const OVERRIDE {
			return m_object->getTableName();
		}
		void execute(std::string &query, const boost::shared_ptr<MySql::Connection> &conn) const OVERRIDE {
			PROFILE_ME;

			m_object->syncGenerateSql(query, m_toReplace);
			LOG_POSEIDON_DEBUG("Executing SQL in ", m_object->getTableName(), ": query = ", query);
			conn->executeSql(query);
		}
	};

	class LoadOperation : public OperationBase {
	private:
		const boost::shared_ptr<MySql::ObjectBase> m_object;
		const std::string m_query;

	public:
		LoadOperation(boost::shared_ptr<MySql::Promise> promise,
			boost::shared_ptr<MySql::ObjectBase> object, std::string query)
			: OperationBase(STD_MOVE(promise))
			, m_object(STD_MOVE(object)), m_query(STD_MOVE(query))
		{
		}

	protected:
		boost::shared_ptr<const MySql::ObjectBase> getCombinableObject() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *getTableName() const OVERRIDE {
			return m_object->getTableName();
		}
		void execute(std::string &query, const boost::shared_ptr<MySql::Connection> &conn) const OVERRIDE {
			PROFILE_ME;

			query = m_query;
			LOG_POSEIDON_INFO("MySQL load: table = ", m_object->getTableName(), ", query = ", query);
			conn->executeSql(query);

			if(!conn->fetchRow()){
				DEBUG_THROW(MySql::Exception, 99999, sslit("No rows returned"));
			}
			m_object->disableAutoSaving();
			m_object->syncFetch(conn);
			m_object->enableAutoSaving();
		}
	};

	class BatchLoadOperation : public OperationBase {
	private:
		boost::shared_ptr<std::deque<boost::shared_ptr<MySql::ObjectBase> > > m_container;
		boost::shared_ptr<MySql::ObjectBase> (*const m_factory)();
		const char *const m_tableHint;
		const std::string m_query;

	public:
		BatchLoadOperation(boost::shared_ptr<MySql::Promise> promise,
			boost::shared_ptr<std::deque<boost::shared_ptr<MySql::ObjectBase> > > container,
			boost::shared_ptr<MySql::ObjectBase> (*factory)(), const char *tableHint, std::string query)
			: OperationBase(STD_MOVE(promise))
			, m_container(STD_MOVE(container)), m_factory(factory), m_tableHint(tableHint), m_query(STD_MOVE(query))
		{
		}

	protected:
		boost::shared_ptr<const MySql::ObjectBase> getCombinableObject() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *getTableName() const OVERRIDE {
			return m_tableHint;
		}
		void execute(std::string &query, const boost::shared_ptr<MySql::Connection> &conn) const OVERRIDE {
			PROFILE_ME;

			query = m_query;
			LOG_POSEIDON_INFO("MySQL batch load: tableHint = ", m_tableHint, ", query = ", query);
			conn->executeSql(query);

			while(conn->fetchRow()){
				AUTO(object, (*m_factory)());
				object->syncFetch(conn);
				object->enableAutoSaving();
				m_container->push_back(STD_MOVE(object));
			}
		}
	};

	const char UNKNOWN_EXCEPTION[] = "Unknown exception";

	class MySqlThread : NONCOPYABLE {
	private:
		struct OperationQueueElement {
			boost::shared_ptr<OperationBase> operation;
			boost::uint64_t dueTime;
			std::size_t retryCount;

			OperationQueueElement(boost::shared_ptr<OperationBase> operation_, boost::uint64_t dueTime_)
				: operation(STD_MOVE(operation_)), dueTime(dueTime_), retryCount(0)
			{
			}
		};

		class WorkingTimeAccumulator : NONCOPYABLE {
		private:
			MySqlThread *const m_owner;
			const char *const m_table;

		public:
			WorkingTimeAccumulator(MySqlThread *owner, const char *table)
				: m_owner(owner), m_table(table)
			{
				m_owner->accumulateTimeForTable("");
			}
			~WorkingTimeAccumulator(){
				m_owner->accumulateTimeForTable(m_table);
			}
		};

		struct TableNameComparator {
			bool operator()(const char *lhs, const char *rhs) const {
				return std::strcmp(lhs, rhs) < 0;
			}
		};

	private:
		Thread m_thread;
		volatile bool m_running;

		mutable Mutex m_mutex;
		mutable ConditionVariable m_newOperation;
		volatile bool m_urgent; // 无视延迟写入，一次性处理队列中所有操作。
		std::deque<OperationQueueElement> m_queue;

		// 性能统计。
		mutable Mutex m_profileMutex;
		double m_profileFlushedTime;
		std::map<const char *, unsigned long long, TableNameComparator> m_profile;

	public:
		MySqlThread()
			: m_running(false)
			, m_urgent(false)
			, m_profileFlushedTime(getHiResMonoClock())
		{
		}

	private:
		void accumulateTimeForTable(const char *table) NOEXCEPT {
			const AUTO(now, getHiResMonoClock());

			const Mutex::UniqueLock lock(m_profileMutex);
			try {
				m_profile[table] += (now - m_profileFlushedTime) * 1e6;
			} catch(...){
			}
			m_profileFlushedTime = now;
		}

		void threadProc(){
			PROFILE_ME;
			LOG_POSEIDON_INFO("MySQL thread started.");

			const MySql::ThreadContext threadContext;
			boost::shared_ptr<MySql::Connection> conn;

			for(;;){
				accumulateTimeForTable("");

				while(!conn){
					LOG_POSEIDON_INFO("Connecting to MySQL server...");

					try {
						conn = MySqlDaemon::createConnection();
						LOG_POSEIDON_INFO("Successfully connected to MySQL server.");
					} catch(std::exception &e){
						LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());

						::timespec req;
						req.tv_sec = (::time_t)(g_reconnDelay / 1000);
						req.tv_nsec = (long)(g_reconnDelay % 1000) * 1000 * 1000;
						::nanosleep(&req, NULLPTR);
					}
				}

				try {
					reallyPumpOperations(conn);
				} catch(MySql::Exception &e){
					LOG_POSEIDON_WARNING("MySql::Exception thrown: code = ", e.code(), ", what = ", e.what());
					conn.reset();
				} catch(std::exception &e){
					LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
					conn.reset();
				} catch(...){
					LOG_POSEIDON_WARNING("Unknown exception thrown");
					conn.reset();
				}
				if(!conn){
					continue;
				}

				Mutex::UniqueLock lock(m_mutex);
				if(!atomicLoad(m_running, ATOMIC_CONSUME) && m_queue.empty()){
					break;
				}
				m_newOperation.timedWait(lock, 100);
			}

			LOG_POSEIDON_INFO("MySQL thread stopped.");
		}

		void reallyPumpOperations(const boost::shared_ptr<MySql::Connection> &conn){
			PROFILE_ME;

			const AUTO(now, getFastMonoClock());

			for(;;){
				OperationQueueElement *elem;
				{
					const Mutex::UniqueLock lock(m_mutex);
					if(m_queue.empty()){
						atomicStore(m_urgent, false, ATOMIC_RELAXED);
						break;
					}
					if(!atomicLoad(m_urgent, ATOMIC_CONSUME) && (now < m_queue.front().dueTime)){
						break;
					}
					elem = &m_queue.front();
				}

				const AUTO(combinableObject, elem->operation->getCombinableObject());
				if(!combinableObject || (combinableObject->getCombinedWriteStamp() == elem)){
					boost::exception_ptr except;

					long errCode = 0;
					char message[4096];
					std::size_t messageLen = 0;

					std::string query;

					try {
						const WorkingTimeAccumulator profiler(this, elem->operation->getTableName());
						elem->operation->execute(query, conn);
					} catch(MySql::Exception &e){
						LOG_POSEIDON_WARNING("MySql::Exception thrown: code = ", e.code(), ", what = ", e.what());
						// except = boost::current_exception();
						except = boost::copy_exception(e);

						errCode = e.code();
						messageLen = std::min(std::strlen(e.what()), sizeof(message));
						std::memcpy(message, e.what(), messageLen);
					} catch(std::exception &e){
						LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
						// except = boost::current_exception();
						except = boost::copy_exception(e);

						errCode = 99999;
						messageLen = std::min(std::strlen(e.what()), sizeof(message));
						std::memcpy(message, e.what(), messageLen);
					} catch(...){
						LOG_POSEIDON_WARNING("Unknown exception thrown");
						except = boost::current_exception();

						errCode = 99999;
						messageLen = sizeof(UNKNOWN_EXCEPTION) - 1;
						std::memcpy(message, UNKNOWN_EXCEPTION, sizeof(UNKNOWN_EXCEPTION));
					}

					const AUTO(promise, elem->operation->getPromise());
					if(except){
						const AUTO(retryCount, ++elem->retryCount);
						if(retryCount < g_maxRetryCount){
							LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
								"Going to retry MySQL operation: retryCount = ", retryCount);
							elem->dueTime = now + (g_retryInitDelay << retryCount);
							boost::rethrow_exception(except);
						}

						LOG_POSEIDON_ERROR("Max retry count exceeded.");
						dumpSql(query, errCode, message, messageLen);
						promise->setException(except);
					} else {
						promise->setSuccess();
					}
				}

				const Mutex::UniqueLock lock(m_mutex);
				m_queue.pop_front();
			}
		}

		void dumpSql(const std::string &query, long errCode, const char *message, std::size_t messageLen){
			PROFILE_ME;

			if(g_dumpDir.empty()){
				LOG_POSEIDON_WARNING("MySQL dump is disabled.");
				return;
			}

			const AUTO(localNow, getLocalTime());
			const AUTO(dt, breakDownTime(localNow));
			char temp[256];
			unsigned len = (unsigned)std::sprintf(temp, "%04u-%02u-%02u %05u", dt.yr, dt.mon, dt.day, (unsigned)::getpid());
			std::string dumpPath;
			dumpPath.assign(g_dumpDir);
			dumpPath.push_back('/');
			dumpPath.append(temp, len);
			dumpPath.append(".log");

			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Creating SQL dump file: ", dumpPath);
			UniqueFile dumpFile;
			if(!dumpFile.reset(::open(dumpPath.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644))){
				const int errCode = errno;
				LOG_POSEIDON_FATAL("Error creating SQL dump file: dumpPath = ", dumpPath,
					", errno = ", errCode, ", desc = ", getErrorDesc(errCode));
				std::abort();
			}

			LOG_POSEIDON_INFO("Writing MySQL dump...");
			std::string dump;
			dump.reserve(1024);
			dump.append("-- Time = ");
			len = formatTime(temp, sizeof(temp), localNow, false);
			dump.append(temp, len);
			dump.append(", Error code = ");
			len = (unsigned)std::sprintf(temp, "%ld", errCode);
			dump.append(temp, len);
			dump.append(", Description = ");
			dump.append(message, messageLen);
			dump.append("\n");
			dump.append(query);
			dump.append(";\n\n");

			const Mutex::UniqueLock lock(g_dumpMutex);
			std::size_t total = 0;
			do {
				::ssize_t written = ::write(dumpFile.get(), dump.data() + total, dump.size() - total);
				if(written <= 0){
					break;
				}
				total += static_cast<std::size_t>(written);
			} while(total < dump.size());
		}

	public:
		void start(){
			Thread(boost::bind(&MySqlThread::threadProc, this), " D  ").swap(m_thread);
			atomicStore(m_running, true, ATOMIC_RELEASE);
		}
		void stop(){
			atomicStore(m_running, false, ATOMIC_RELEASE);
		}
		void safeJoin(){
			waitTillIdle();

			if(m_thread.joinable()){
				m_thread.join();
			}
		}

		void getProfile(std::vector<std::pair<const char *, unsigned long long> > &ret) const {
			const Mutex::UniqueLock lock(m_profileMutex);
			ret.reserve(ret.size() + m_profile.size());
			for(AUTO(it, m_profile.begin()); it != m_profile.end(); ++it){
				ret.push_back(std::make_pair(it->first, it->second));
			}
		}

		void waitTillIdle(){
			for(;;){
				std::size_t pendingObjects;
				{
					const Mutex::UniqueLock lock(m_mutex);
					pendingObjects = m_queue.size();
					if(pendingObjects == 0){
						break;
					}
					atomicStore(m_urgent, true, ATOMIC_RELEASE);
					m_newOperation.signal();
				}
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "There are ", pendingObjects, " object(s) in my queue.");

				::timespec req;
				req.tv_sec = 0;
				req.tv_nsec = 500 * 1000 * 1000;
				::nanosleep(&req, NULLPTR);
			}
		}

		void addOperation(boost::shared_ptr<OperationBase> operation, bool urgent){
			PROFILE_ME;

			if(!atomicLoad(m_running, ATOMIC_CONSUME)){
				LOG_POSEIDON_ERROR("MySQL thread is being shut down.");
				DEBUG_THROW(Exception, sslit("MySQL thread is being shut down"));
			}

			const AUTO(combinableObject, operation->getCombinableObject());

			AUTO(dueTime, getFastMonoClock());
			// 有紧急操作时无视写入延迟，这个逻辑不在这里处理。
			// if(combinableObject && !urgent) // 这个看似合理但是实际是错的。
			if(combinableObject || urgent){ // 确保紧急操作排在其他所有操作之后。
				dueTime += g_saveDelay;
			}

			const Mutex::UniqueLock lock(m_mutex);
			m_queue.push_back(OperationQueueElement(STD_MOVE(operation), dueTime));
			OperationQueueElement *const elem = &m_queue.back();
			if(combinableObject){
				combinableObject->setCombinedWriteStamp(elem);
			}
			if(urgent){
				atomicStore(m_urgent, true, ATOMIC_RELEASE);
			}
			m_newOperation.signal();
		}
	};

	volatile bool g_running = false;
	boost::container::vector<MySqlThread> g_threads;

	void submitOperation(const char *table, boost::shared_ptr<OperationBase> operation, bool urgent){
		if(g_threads.empty()){
			DEBUG_THROW(Exception, sslit("No MySQL thread is running"));
		}

		// http://www.isthe.com/chongo/tech/comp/fnv/
		std::size_t hash;
		if(sizeof(std::size_t) < 8){
			hash = 2166136261u;
			const char *p = table;
			while(*p){
				hash ^= static_cast<unsigned char>(*p);
				hash *= 16777619u;
				++p;
			}
		} else {
			hash = 14695981039346656037u;
			const char *p = table;
			while(*p){
				hash ^= static_cast<unsigned char>(*p);
				hash *= 1099511628211u;
				++p;
			}
		}
		const AUTO(threadIndex, hash % g_threads.size());
		LOG_POSEIDON_DEBUG("Assigning MySQL table `", table, "` to thread ", threadIndex);

		g_threads.at(threadIndex).addOperation(STD_MOVE(operation), urgent);
	}
}

void MySqlDaemon::start(){
	if(atomicExchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting MySQL daemon...");

	MainConfig::get(g_serverAddr, "mysql_server_addr");
	LOG_POSEIDON_DEBUG("MySQL server addr = ", g_serverAddr);

	MainConfig::get(g_serverPort, "mysql_server_port");
	LOG_POSEIDON_DEBUG("MySQL server port = ", g_serverPort);

	MainConfig::get(g_username, "mysql_username");
	LOG_POSEIDON_DEBUG("MySQL username = ", g_username);

	MainConfig::get(g_password, "mysql_password");
	LOG_POSEIDON_DEBUG("MySQL password = ", g_password);

	MainConfig::get(g_schema, "mysql_schema");
	LOG_POSEIDON_DEBUG("MySQL schema = ", g_schema);

	MainConfig::get(g_useSsl, "mysql_use_ssl");
	LOG_POSEIDON_DEBUG("MySQL use ssl = ", g_useSsl);

	MainConfig::get(g_charset, "mysql_charset");
	LOG_POSEIDON_DEBUG("MySQL charset = ", g_charset);

	MainConfig::get(g_dumpDir, "mysql_dump_dir");
	LOG_POSEIDON_DEBUG("MySQL dump dir = ", g_dumpDir);

	MainConfig::get(g_maxThreads, "mysql_max_threads");
	LOG_POSEIDON_DEBUG("MySQL max threads = ", g_maxThreads);

	MainConfig::get(g_saveDelay, "mysql_save_delay");
	LOG_POSEIDON_DEBUG("MySQL save delay = ", g_saveDelay);

	MainConfig::get(g_reconnDelay, "mysql_reconn_delay");
	LOG_POSEIDON_DEBUG("MySQL reconnect delay = ", g_reconnDelay);

	MainConfig::get(g_maxRetryCount, "mysql_max_retry_count");
	LOG_POSEIDON_DEBUG("MySQL max retry count = ", g_maxRetryCount);

	MainConfig::get(g_retryInitDelay, "mysql_retry_init_delay");
	LOG_POSEIDON_DEBUG("MySQL retry init delay = ", g_retryInitDelay);

	const AUTO(threadCount, std::max<std::size_t>(g_maxThreads, 1));
	boost::container::vector<MySqlThread> threads(threadCount);
	for(std::size_t i = 0; i < threads.size(); ++i){
		LOG_POSEIDON_INFO("Creating MySQL thread ", i);
		threads.at(i).start();
	}
	g_threads.swap(threads);

	if(!g_dumpDir.empty()){
		const AUTO(placeholderPath, g_dumpDir + "/placeholder");
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Checking whether MySQL dump directory is writeable: testPath = ", placeholderPath);
		UniqueFile dumpFile;
		if(!dumpFile.reset(::open(placeholderPath.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0644))){
			const int errCode = errno;
			LOG_POSEIDON_FATAL("Could not create placeholder file: placeholderPath = ", placeholderPath, ", errno = ", errCode);
			std::abort();
		}
	}

	LOG_POSEIDON_INFO("MySQL daemon started.");
}
void MySqlDaemon::stop(){
	if(atomicExchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping MySQL daemon...");

	boost::container::vector<MySqlThread> threads;
	threads.swap(g_threads);
	for(std::size_t i = 0; i < threads.size(); ++i){
		LOG_POSEIDON_INFO("Stopping MySQL thread ", i);
		threads.at(i).stop();
	}
	for(std::size_t i = 0; i < threads.size(); ++i){
		LOG_POSEIDON_INFO("Waiting for MySQL thread ", i, " to terminate...");
		threads.at(i).safeJoin();
	}

	LOG_POSEIDON_INFO("MySQL daemon stopped.");
}

boost::shared_ptr<MySql::Connection> MySqlDaemon::createConnection(){
	return MySql::Connection::create(g_serverAddr, g_serverPort, g_username, g_password, g_schema, g_useSsl, g_charset);
}

std::vector<MySqlDaemon::SnapshotElement> MySqlDaemon::snapshot(){
	std::vector<SnapshotElement> ret;
	ret.reserve(100);

	std::vector<std::pair<const char *, unsigned long long> > threadProfile;
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		g_threads.at(i).getProfile(threadProfile);
		ret.reserve(ret.size() + threadProfile.size());
		for(AUTO(it, threadProfile.begin()); it != threadProfile.end(); ++it){
			ret.push_back(VAL_INIT);
			AUTO_REF(elem, ret.back());
			elem.thread = i;
			elem.table = it->first;
			elem.nsTotal = it->second;
		}
	}

	return ret;
}

void MySqlDaemon::waitForAllAsyncOperations(){
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		g_threads.at(i).waitTillIdle();
	}
}

boost::shared_ptr<const MySql::Promise> MySqlDaemon::enqueueForSaving(
	boost::shared_ptr<const MySql::ObjectBase> object, bool toReplace, bool urgent)
{
	AUTO(promise, boost::make_shared<MySql::Promise>());
	const char *const tableName = object->getTableName();
	submitOperation(tableName,
		boost::make_shared<SaveOperation>(promise, STD_MOVE(object), toReplace),
		urgent);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const MySql::Promise> MySqlDaemon::enqueueForLoading(
	boost::shared_ptr<MySql::ObjectBase> object, std::string query)
{
	AUTO(promise, boost::make_shared<MySql::Promise>());
	const char *const tableName = object->getTableName();
	submitOperation(tableName,
		boost::make_shared<LoadOperation>(promise, STD_MOVE(object), STD_MOVE(query)),
		true);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const MySql::Promise> MySqlDaemon::enqueueForBatchLoading(
	boost::shared_ptr<std::deque<boost::shared_ptr<MySql::ObjectBase> > > sink,
	boost::shared_ptr<MySql::ObjectBase> (*factory)(), const char *tableHint, std::string query)
{
	AUTO(promise, boost::make_shared<MySql::Promise>());
	const char *const tableName = tableHint;
	submitOperation(tableName,
		boost::make_shared<BatchLoadOperation>(promise, STD_MOVE(sink), factory, tableHint, STD_MOVE(query)),
		true);
	return STD_MOVE_IDN(promise);
}

}
