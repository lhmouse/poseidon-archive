#include "../../precompiled.hpp"
#include "mysql_daemon.hpp"
#include <list>
#include <boost/thread.hpp>
#include <boost/scoped_ptr.hpp>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <mysql/mysql.h>
#include <unistd.h>
#include "config_file.hpp"
#include "../mysql/object_base.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../utilities.hpp"
using namespace Poseidon;

namespace {

std::string g_databaseServer			= "tcp://localhost:3306";
std::string g_databaseUsername			= "root";
std::string g_databasePassword			= "root";
std::string g_databaseName				= "test";

std::size_t g_databaseSaveDelay			= 5000;
std::size_t g_databaseMaxReconnDelay	= 60000;

class AsyncLoadCallbackJob : public JobBase {
private:
	MySqlAsyncLoadCallback m_callback;
	boost::shared_ptr<MySqlObjectBase> m_object;

public:
	AsyncLoadCallbackJob(MySqlAsyncLoadCallback callback,
		boost::shared_ptr<MySqlObjectBase> object)
	{
		m_callback.swap(callback);
		m_object.swap(object);
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

	void swap(AsyncSaveItem &rhs) throw() {
		object.swap(rhs.object);
		std::swap(timeStamp, rhs.timeStamp);
	}
};

struct AsyncLoadItem {
	boost::shared_ptr<MySqlObjectBase> object;
	std::string filter;
	MySqlAsyncLoadCallback callback;

	void swap(AsyncLoadItem &rhs) throw() {
		object.swap(rhs.object);
		filter.swap(rhs.filter);
		callback.swap(rhs.callback);
	}
};

volatile bool g_running = false;
boost::thread g_thread;

boost::mutex g_mutex;
std::list<AsyncSaveItem> g_saveQueue;
std::list<AsyncSaveItem> g_savePool;
std::list<AsyncLoadItem> g_loadQueue;
std::list<AsyncLoadItem> g_loadPool;
boost::condition_variable g_newObjectAvail;
boost::condition_variable g_queueEmpty;

void getMySqlConnection(boost::scoped_ptr<sql::Connection> &connection){
	LOG_INFO("Connecting to MySQL server...");

	std::size_t reconnectDelay = 0;
	for(;;){
		try {
			connection.reset(::get_driver_instance()->connect(
				g_databaseServer, g_databaseUsername, g_databasePassword));
			connection->setSchema(g_databaseName);
		} catch(sql::SQLException &e){
			LOG_ERROR("Error connecting to MySQL server: code = ", e.getErrorCode(),
				", state = ", e.getSQLState(), ", what = ", e.what());
			connection.reset();
		}
		if(connection){
			break;
		}
		if(reconnectDelay == 0){
			reconnectDelay = 1;
		} else {
			LOG_INFO("Will retry after ", reconnectDelay, " milliseconds.");
			::usleep(reconnectDelay * 1000);

			reconnectDelay <<= 1;
			if(reconnectDelay > g_databaseMaxReconnDelay){
				reconnectDelay = g_databaseMaxReconnDelay;
			}
		}
	}

	LOG_INFO("Successfully connected to MySQL server.");
}

void daemonLoop(){
	boost::scoped_ptr<sql::Connection> connection;
	for(;;){
		bool discardConnection = false;

		try {
			if(!connection){
				getMySqlConnection(connection);
			}

			AsyncSaveItem asi;
			AsyncLoadItem ali;
			{
				boost::mutex::scoped_lock lock(g_mutex);
				for(;;){
					if(!g_saveQueue.empty()){
						AUTO_REF(head, g_saveQueue.front());
						if(head.timeStamp > getMonoClock()){
							goto skip;
						}
						if(head.object->getContext() != &head){
							AsyncSaveItem().swap(head);
						} else {
							asi.swap(head);
						}
						g_savePool.splice(g_savePool.begin(), g_saveQueue, g_saveQueue.begin());
						if(!asi.object){
							goto skip;
						}
						break;
					}
				skip:
					if(!g_loadQueue.empty()){
						ali.swap(g_loadQueue.front());
						g_loadPool.splice(g_loadPool.begin(), g_loadQueue, g_loadQueue.begin());
						break;
					}
					if(!atomicLoad(g_running)){
						break;
					}
					g_newObjectAvail.timed_wait(lock, boost::posix_time::seconds(1));
				}
			}
			if(asi.object){
				asi.object->syncSave(connection.get());
			} else if(ali.object){
				ali.object->syncLoad(connection.get(), ali.filter.c_str());
				ali.object->enableAutoSaving();

				if(ali.callback){
					boost::make_shared<AsyncLoadCallbackJob>(
						boost::ref(ali.callback), boost::ref(ali.object))->pend();
				}
			} else {
				break;
			}
		} catch(sql::SQLException &e){
			LOG_ERROR("SQLException thrown in MySQL daemon: code = ", e.getErrorCode(),
				", state = ", e.getSQLState(), ", what = ", e.what());
			discardConnection = true;
		} catch(std::exception &e){
			LOG_ERROR("std::exception thrown in MySQL daemon: what = ", e.what());
			discardConnection = true;
		} catch(...){
			LOG_ERROR("Unknown exception thrown in MySQL daemon.");
			discardConnection = true;
		}

		if(discardConnection){
			LOG_INFO("The connection was left in an indeterminate state. Discard it.");
			connection.reset();
		}
	}
}

void threadProc(){
	PROFILE_ME;
	Logger::setThreadTag(Logger::TAG_MYSQL);
	LOG_INFO("MySQL daemon started.");

	daemonLoop();
	::mysql_thread_end();

	LOG_INFO("MySQL daemon stopped.");
}

}

void MySqlDaemon::start(){
	if(atomicExchange(g_running, true) != false){
		LOG_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_INFO("Starting MySQL daemon...");

	ConfigFile::get(g_databaseServer, "database_server");
	LOG_DEBUG("MySQL server = ", g_databaseServer);

	ConfigFile::get(g_databaseUsername, "database_username");
	LOG_DEBUG("MySQL username = ", g_databaseUsername);

	ConfigFile::get(g_databasePassword, "database_password");
	LOG_DEBUG("MySQL password = ", g_databasePassword);

	ConfigFile::get(g_databaseName, "database_name");
	LOG_DEBUG("MySQL database name = ", g_databaseName);

	ConfigFile::get(g_databaseSaveDelay, "database_save_delay");
	LOG_DEBUG("MySQL save delay = ", g_databaseSaveDelay);

	ConfigFile::get(g_databaseMaxReconnDelay, "database_max_reconn_delay");
	LOG_DEBUG("MySQL max reconnect delay = ", g_databaseMaxReconnDelay);

	boost::thread(threadProc).swap(g_thread);
}
void MySqlDaemon::stop(){
	LOG_INFO("Stopping MySQL daemon...");

	atomicStore(g_running, false);
	{
		const boost::mutex::scoped_lock lock(g_mutex);
		g_newObjectAvail.notify_all();
	}
	if(g_thread.joinable()){
		g_thread.join();
	}
}

void MySqlDaemon::waitForAllAsyncOperations(){
	boost::mutex::scoped_lock lock(g_mutex);
	while(!(g_saveQueue.empty() && g_loadQueue.empty())){
		g_queueEmpty.wait(lock);
	}
}

void MySqlDaemon::pendForSaving(boost::shared_ptr<const MySqlObjectBase> object){
	const boost::mutex::scoped_lock lock(g_mutex);
	if(g_savePool.empty()){
		g_savePool.push_front(AsyncSaveItem());
	}
	g_saveQueue.splice(g_saveQueue.end(), g_savePool, g_savePool.begin());

	AUTO_REF(asi, g_saveQueue.back());
	asi.object.swap(object);
	asi.timeStamp = getMonoClock() + g_databaseSaveDelay * 1000;
	asi.object->setContext(&asi);

	g_newObjectAvail.notify_all();
}
void MySqlDaemon::pendForLoading(boost::shared_ptr<MySqlObjectBase> object,
	std::string filter, MySqlAsyncLoadCallback callback)
{
	const boost::mutex::scoped_lock lock(g_mutex);
	if(g_loadPool.empty()){
		g_loadPool.push_front(AsyncLoadItem());
	}
	g_loadQueue.splice(g_loadQueue.end(), g_loadPool, g_loadPool.begin());

	AUTO_REF(ali, g_loadQueue.back());
	ali.object.swap(object);
	ali.filter.swap(filter);
	ali.callback.swap(callback);

	g_newObjectAvail.notify_all();
}
