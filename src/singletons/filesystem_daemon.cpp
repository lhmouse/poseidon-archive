// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "filesystem_daemon.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "../thread.hpp"
#include "../mutex.hpp"
#include "../condition_variable.hpp"
#include "../atomic.hpp"
#include "../system_exception.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../job_promise.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace {
	void real_load_file(std::string &data, const std::string &path, bool throws_if_does_not_exist){
		int flags = O_RDONLY;
		UniqueFile file;
		if(!file.reset(::open(path.c_str(), flags))){
			const int err_code = errno;
			if(!throws_if_does_not_exist && (err_code == ENOENT)){
				return;
			}
			LOG_POSEIDON_DEBUG("Failed to load file: path = ", path, ", err_code = ", err_code);
			DEBUG_THROW(SystemException, err_code);
		}

		std::size_t bytes_read = 0;
		for(;;){
			char temp[16384];
			const ::ssize_t result = ::read(file.get(), temp, sizeof(temp));
			if(result == 0){
				break;
			}
			if(result < 0){
				const int err_code = errno;
				LOG_POSEIDON_ERROR("Error loading file: path = ", path, ", err_code = ", err_code);
				DEBUG_THROW(SystemException, err_code);
			}
			const std::size_t avail = static_cast<std::size_t>(result);
			data.append(temp, avail);
			bytes_read += avail;
		}
		LOG_POSEIDON_DEBUG("Finished loading file: path = ", path, ", bytes_read = ", bytes_read);
	}
	void real_save_file(const std::string &data, const std::string &path, bool appends, bool forces_creation){
		int flags = O_CREAT | O_WRONLY;
		if(appends){
			flags |= O_APPEND;
		} else {
			flags |= O_TRUNC;
		}
		if(forces_creation){
			flags |= O_EXCL;
		}
		UniqueFile file;
		if(!file.reset(::open(path.c_str(), flags, static_cast< ::mode_t>(0666)))){
			const int err_code = errno;
			LOG_POSEIDON_DEBUG("Failed to save file: path = ", path, ", err_code = ", err_code);
			DEBUG_THROW(SystemException, err_code);
		}

		std::size_t bytes_written = 0;
		for(;;){
			char temp[16384];
			const std::size_t avail = data.copy(temp, sizeof(temp), bytes_written);
			if(avail == 0){
				break;
			}
			const ::ssize_t result = ::write(file.get(), temp, avail);
			if(result < 0){
				const int err_code = errno;
				LOG_POSEIDON_ERROR("Error saving file: path = ", path, ", err_code = ", err_code);
				DEBUG_THROW(SystemException, err_code);
			}
			bytes_written += avail;
		}
		LOG_POSEIDON_DEBUG("Finished saving file: path = ", path, ", bytes_written = ", bytes_written);
	}
	void real_remove_file(const std::string &path, bool throws_if_does_not_exist){
		if(!::unlink(path.c_str())){
			const int err_code = errno;
			if(!throws_if_does_not_exist && (err_code == ENOENT)){
				return;
			}
			LOG_POSEIDON_DEBUG("Failed to remove file: path = ", path, ", err_code = ", err_code);
			DEBUG_THROW(SystemException, err_code);
		}
	}

	class OperationBase : NONCOPYABLE {
	public:
		virtual ~OperationBase(){
		}

	public:
		virtual void execute() const = 0;
	};

	class LoadOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const boost::shared_ptr<std::string> m_data;
		const std::string m_path;
		const bool m_throws_if_does_not_exist;

	public:
		LoadOperation(boost::shared_ptr<JobPromise> promise,
			boost::shared_ptr<std::string> data, std::string path, bool throws_if_does_not_exist)
			: m_promise(STD_MOVE(promise))
			, m_data(STD_MOVE(data)), m_path(STD_MOVE(path)), m_throws_if_does_not_exist(throws_if_does_not_exist)
		{
		}

	public:
		void execute() const OVERRIDE {
			if(m_promise.unique()){
				LOG_POSEIDON_DEBUG("Discarding isolated loading operation: path = ", m_path);
				return;
			}

			try {
				real_load_file(*m_data, m_path, m_throws_if_does_not_exist);
				m_promise->set_success();
			} catch(SystemException &e){
				LOG_POSEIDON_INFO("SystemException thrown: what = ", e.what(), ", code = ", e.get_code());
#ifdef POSEIDON_CXX11
				m_promise->set_exception(std::current_exception());
#else
				m_promise->set_exception(boost::copy_exception(e));
#endif
			} catch(std::exception &e){
				LOG_POSEIDON_INFO("std::exception thrown: what = ", e.what());
#ifdef POSEIDON_CXX11
				m_promise->set_exception(std::current_exception());
#else
				m_promise->set_exception(boost::copy_exception(std::runtime_error(e.what())));
#endif
			}
		}
	};

	class SaveOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const std::string m_data;
		const std::string m_path;
		const bool m_appends;
		const bool m_forces_creation;

	public:
		SaveOperation(boost::shared_ptr<JobPromise> promise,
			std::string data, std::string path, bool appends, bool forces_creation)
			: m_promise(STD_MOVE(promise))
			, m_data(STD_MOVE(data)), m_path(STD_MOVE(path)), m_appends(appends), m_forces_creation(forces_creation)
		{
		}

	public:
		void execute() const OVERRIDE {
			try {
				real_save_file(m_data, m_path, m_appends, m_forces_creation);
				m_promise->set_success();
			} catch(SystemException &e){
				LOG_POSEIDON_INFO("SystemException thrown: what = ", e.what(), ", code = ", e.get_code());
#ifdef POSEIDON_CXX11
				m_promise->set_exception(std::current_exception());
#else
				m_promise->set_exception(boost::copy_exception(e));
#endif
			} catch(std::exception &e){
				LOG_POSEIDON_INFO("std::exception thrown: what = ", e.what());
#ifdef POSEIDON_CXX11
				m_promise->set_exception(std::current_exception());
#else
				m_promise->set_exception(boost::copy_exception(std::runtime_error(e.what())));
#endif
			}
		}
	};

	class RemoveOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const std::string m_path;
		const bool m_throws_if_does_not_exist;

	public:
		RemoveOperation(boost::shared_ptr<JobPromise> promise,
			std::string path, bool throws_if_does_not_exist)
			: m_promise(STD_MOVE(promise))
			, m_path(STD_MOVE(path)), m_throws_if_does_not_exist(throws_if_does_not_exist)
		{
		}

	public:
		void execute() const OVERRIDE {
			try {
				real_remove_file(m_path, m_throws_if_does_not_exist);
				m_promise->set_success();
			} catch(SystemException &e){
				LOG_POSEIDON_INFO("SystemException thrown: what = ", e.what(), ", code = ", e.get_code());
#ifdef POSEIDON_CXX11
				m_promise->set_exception(std::current_exception());
#else
				m_promise->set_exception(boost::copy_exception(e));
#endif
			} catch(std::exception &e){
				LOG_POSEIDON_INFO("std::exception thrown: what = ", e.what());
#ifdef POSEIDON_CXX11
				m_promise->set_exception(std::current_exception());
#else
				m_promise->set_exception(boost::copy_exception(std::runtime_error(e.what())));
#endif
			}
		}
	};

	volatile bool g_running = false;
	Thread g_thread;

	Mutex g_mutex;
	ConditionVariable g_new_operation;
	boost::container::deque<boost::shared_ptr<OperationBase> > g_operations;

	bool pump_one_element() NOEXCEPT {
		PROFILE_ME;

		boost::shared_ptr<OperationBase> operation;
		{
			const Mutex::UniqueLock lock(g_mutex);
			if(!g_operations.empty()){
				operation = g_operations.front();
			}
		}
		if(!operation){
			return false;
		}

		try {
			operation->execute();
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown.");
		}
		const Mutex::UniqueLock lock(g_mutex);
		g_operations.pop_front();
		return true;
	}

	void daemon_loop(){
		PROFILE_ME;

		for(;;){
			while(pump_one_element()){
				// noop
			}

			if(!atomic_load(g_running, ATOMIC_CONSUME)){
				break;
			}

			Mutex::UniqueLock lock(g_mutex);
			g_new_operation.timed_wait(lock, 100);
		}
	}

	void thread_proc(){
		PROFILE_ME;
		LOG_POSEIDON_INFO("Filesystem daemon started.");

		daemon_loop();

		LOG_POSEIDON_INFO("Filesystem daemon stopped.");
	}
}

void FilesystemDaemon::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting Filesystem daemon...");

	Thread(thread_proc, " F  ").swap(g_thread);
}
void FilesystemDaemon::stop(){
	if(atomic_exchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping Filesystem daemon...");

	if(g_thread.joinable()){
		g_thread.join();
	}
	g_operations.clear();
}

void FilesystemDaemon::load(std::string &data, const std::string &path, bool throws_if_does_not_exist){
	PROFILE_ME;

	real_load_file(data, path, throws_if_does_not_exist);
}
void FilesystemDaemon::save(std::string data, const std::string &path, bool appends, bool forces_creation){
	PROFILE_ME;

	real_save_file(STD_MOVE(data), path, appends, forces_creation);
}
void FilesystemDaemon::remove(const std::string &path, bool throws_if_does_not_exist){
	PROFILE_ME;

	real_remove_file(path, throws_if_does_not_exist);
}

boost::shared_ptr<const JobPromise> FilesystemDaemon::enqueue_for_loading(
	boost::shared_ptr<std::string> data, std::string path, bool throws_if_does_not_exist)
{
	PROFILE_ME;

	AUTO(promise, boost::make_shared<JobPromise>());
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_operations.push_back(boost::make_shared<LoadOperation>(
			promise, STD_MOVE(data), STD_MOVE(path), throws_if_does_not_exist));
		g_new_operation.signal();
	}
	return promise;
}
boost::shared_ptr<const JobPromise> FilesystemDaemon::enqueue_for_saving(
	std::string data, std::string path, bool appends, bool forces_creation)
{
	PROFILE_ME;

	AUTO(promise, boost::make_shared<JobPromise>());
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_operations.push_back(boost::make_shared<SaveOperation>(
			promise, STD_MOVE(data), STD_MOVE(path), appends, forces_creation));
		g_new_operation.signal();
	}
	return promise;
}
boost::shared_ptr<const JobPromise> FilesystemDaemon::enqueue_for_removing(
	std::string path, bool throws_if_does_not_exist)
{
	PROFILE_ME;

	AUTO(promise, boost::make_shared<JobPromise>());
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_operations.push_back(boost::make_shared<RemoveOperation>(
			promise, STD_MOVE(path), throws_if_does_not_exist));
		g_new_operation.signal();
	}
	return promise;
}

}
