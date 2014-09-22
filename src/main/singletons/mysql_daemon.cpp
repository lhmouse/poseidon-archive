#include "../../precompiled.hpp"
#include "mysql_daemon.hpp"
#include <list>
#include <boost/thread.hpp>
#include <boost/scoped_ptr.hpp>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <unistd.h>
#include "config_file.hpp"
#include "../mysql/object_base.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
using namespace Poseidon;

namespace {

sql::SQLString g_databaseServer			= "tcp://localhost:3306";
sql::SQLString g_databaseUsername		= "root";
sql::SQLString g_databasePassword		= "root";
sql::SQLString g_databaseName			= "test";

std::size_t g_databaseSaveDelay			= 5000;
std::size_t g_databaseMaxReconnDelay	= 60000;

volatile bool g_daemonRunning = false;
boost::thread g_daemonThread;

boost::mutex g_mutex;

boost::condition_variable g_objectAvail;
boost::condition_variable g_queueEmpty;

void getMySqlConnection(boost::scoped_ptr<sql::Connection> &connection){
	LOG_INFO("Connecting to MySQL server...");

	std::size_t reconnectDelay = 0;
	for(;;){
		try {
			connection.reset(::get_driver_instance()->connect(
				g_databaseServer, g_databaseUsername, g_databasePassword));
			connection->setSchema(g_databaseName);
			break;
		} catch(sql::SQLException &e){
			LOG_ERROR("Error connecting to MySQL server: code = ", e.getErrorCode(),
				", state = ", e.getSQLState(), ", what = ", e.what());
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

	LOG_INFO("Successfully connected.");
}

void threadProc(){
	LOG_INFO("MySQL daemon started.");

	g_databaseServer =
		ConfigFile::get<std::string>("database_server", g_databaseServer);
	LOG_DEBUG("MySQL server = ", g_databaseServer);

	g_databaseUsername =
		ConfigFile::get<std::string>("database_username", g_databaseUsername);
	LOG_DEBUG("MySQL username = ", g_databaseUsername);

	g_databasePassword =
		ConfigFile::get<std::string>("database_password", g_databasePassword);
	LOG_DEBUG("MySQL password = ", g_databasePassword);

	g_databaseName =
		ConfigFile::get<std::string>("database_name", g_databaseName);
	LOG_DEBUG("MySQL database name = ", g_databaseName);

	g_databaseSaveDelay =
		ConfigFile::get<std::size_t>("database_save_delay", g_databaseSaveDelay);
	LOG_DEBUG("MySQL save delay = ", g_databaseSaveDelay);

	g_databaseMaxReconnDelay =
		ConfigFile::get<std::size_t>("database_max_reconn_delay", g_databaseMaxReconnDelay);
	LOG_DEBUG("MySQL max reconnect delay = ", g_databaseMaxReconnDelay);

	boost::scoped_ptr<sql::Connection> connection;

	for(;;){
		try {
			if(!connection){
				getMySqlConnection(connection);
			}
/*
			boost::shared_ptr<const MySqlObjectBase> object;
			{
				boost::mutex::scoped_lock lock(g_mutex);
				for(;;){
					if(!g_queue.empty()){
						object = STD_MOVE(g_queue.front());
						g_queue.pop_front();
						break;
					}
					if(!atomicLoad(g_daemonRunning)){
						break;
					}
					g_objectAvail.wait(lock);
				}
			}
			if(!object){
				break;
			}
*/			//job->perform();
		} catch(sql::SQLException &e){
			LOG_ERROR("SQLException thrown in MySQL daemon: code = ", e.getErrorCode(),
				", state = ", e.getSQLState(), ", what = ", e.what());

			LOG_INFO("The connection was left in an indeterminate state. Free it.");
			connection.reset();
		} catch(Exception &e){
			LOG_ERROR("Exception thrown in MySQL daemon: file = ", e.file(),
				", line = ", e.line(), ", what = ", e.what());
		} catch(std::exception &e){
			LOG_ERROR("std::exception thrown in MySQL daemon: what = ", e.what());
		} catch(...){
			LOG_ERROR("Unknown exception thrown in MySQL daemon.");
		}
	}

	LOG_INFO("MySQL daemon stopped.");
}

}

void MySqlDaemon::start(){
	if(atomicExchange(g_daemonRunning, true) != false){
		LOG_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_INFO("Starting MySQL daemon...");

	boost::thread(threadProc).swap(g_daemonThread);
}
void MySqlDaemon::stop(){
	LOG_INFO("Stopping MySQL daemon...");

	atomicStore(g_daemonRunning, false);
	g_objectAvail.notify_one();
	if(g_daemonThread.joinable()){
		g_daemonThread.join();
	}
}

void MySqlDaemon::waitForAllAsyncOperations(){
/*	boost::mutex::scoped_lock lock(g_mutex);
	while(!g_queue.empty()){
		g_queueEmpty.wait(lock);
	}
*/
}

void MySqlDaemon::pendForSaving(boost::shared_ptr<const MySqlObjectBase> object){
(void)object;
	g_objectAvail.notify_one();
}
void MySqlDaemon::pendForLoading(boost::shared_ptr<MySqlObjectBase> object,
	std::string filter, MySqlAsyncLoadCallback callback)
{
(void)object;
(void)filter;
(void)callback;
	g_objectAvail.notify_one();
}
