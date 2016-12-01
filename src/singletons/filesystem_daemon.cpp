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
	void real_load(StreamBuffer &data, const std::string &path, bool throws_if_does_not_exist){
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
			data.put(temp, avail);
			bytes_read += avail;
		}
		LOG_POSEIDON_DEBUG("Finished loading file: path = ", path, ", bytes_read = ", bytes_read);
	}
	void real_save(StreamBuffer data, const std::string &path, bool appends, bool throws_if_exists){
		int flags = O_CREAT | O_WRONLY;
		if(appends){
			flags |= O_APPEND;
		} else {
			flags |= O_TRUNC;
		}
		if(throws_if_exists){
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
			const std::size_t avail = data.get(temp, sizeof(temp));
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
	void real_remove(const std::string &path, bool throws_if_does_not_exist){
		if(!::unlink(path.c_str())){
			const int err_code = errno;
			if(!throws_if_does_not_exist && (err_code == ENOENT)){
				return;
			}
			LOG_POSEIDON_DEBUG("Failed to remove file: path = ", path, ", err_code = ", err_code);
			DEBUG_THROW(SystemException, err_code);
		}
	}
	void real_rename(const std::string &path, const std::string &new_path){
		if(!::rename(path.c_str(), new_path.c_str())){
			const int err_code = errno;
			LOG_POSEIDON_DEBUG("Failed to rename file: path = ", path, ", err_code = ", err_code);
			DEBUG_THROW(SystemException, err_code);
		}
	}
	void real_mkdir(const std::string &path, bool throws_if_exists){
		if(!::mkdir(path.c_str(), static_cast< ::mode_t>(0777))){
			const int err_code = errno;
			if(!throws_if_exists && (err_code == EEXIST)){
				struct ::stat buf;
				if((::lstat(path.c_str(), &buf) == 0) && S_ISDIR(buf.st_mode)){
					return;
				}
			}
			LOG_POSEIDON_DEBUG("Failed to make directory: path = ", path, ", err_code = ", err_code);
			DEBUG_THROW(SystemException, err_code);
		}
	}
	void real_rmdir(const std::string &path, bool throws_if_does_not_exist){
		if(!::rmdir(path.c_str())){
			const int err_code = errno;
			if(!throws_if_does_not_exist && (err_code == ENOENT)){
				return;
			}
			LOG_POSEIDON_DEBUG("Failed to remove directory: path = ", path, ", err_code = ", err_code);
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
		const boost::shared_ptr<StreamBuffer> m_data;
		const std::string m_path;
		const bool m_throws_if_does_not_exist;

	public:
		LoadOperation(boost::shared_ptr<JobPromise> promise,
			boost::shared_ptr<StreamBuffer> data, std::string path, bool throws_if_does_not_exist)
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
				real_load(*m_data, m_path, m_throws_if_does_not_exist);
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
		const StreamBuffer m_data;
		const std::string m_path;
		const bool m_appends;
		const bool m_throws_if_exists;

	public:
		SaveOperation(boost::shared_ptr<JobPromise> promise,
			StreamBuffer data, std::string path, bool appends, bool throws_if_exists)
			: m_promise(STD_MOVE(promise))
			, m_data(STD_MOVE(data)), m_path(STD_MOVE(path)), m_appends(appends), m_throws_if_exists(throws_if_exists)
		{
		}

	public:
		void execute() const OVERRIDE {
			try {
				real_save(m_data, m_path, m_appends, m_throws_if_exists);
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
				real_remove(m_path, m_throws_if_does_not_exist);
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

	class RenameOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const std::string m_path;
		const std::string m_new_path;

	public:
		RenameOperation(boost::shared_ptr<JobPromise> promise,
			std::string path, std::string new_path)
			: m_promise(STD_MOVE(promise))
			, m_path(STD_MOVE(path)), m_new_path(STD_MOVE(new_path))
		{
		}

	public:
		void execute() const OVERRIDE {
			try {
				real_rename(m_path, m_new_path);
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

	class MkdirOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const std::string m_path;
		const bool m_throws_if_exists;

	public:
		MkdirOperation(boost::shared_ptr<JobPromise> promise,
			std::string path, bool throws_if_exists)
			: m_promise(STD_MOVE(promise))
			, m_path(STD_MOVE(path)), m_throws_if_exists(throws_if_exists)
		{
		}

	public:
		void execute() const OVERRIDE {
			try {
				real_mkdir(m_path, m_throws_if_exists);
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

	class RmdirOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const std::string m_path;
		const bool m_throws_if_does_not_exist;

	public:
		RmdirOperation(boost::shared_ptr<JobPromise> promise,
			std::string path, bool throws_if_does_not_exist)
			: m_promise(STD_MOVE(promise))
			, m_path(STD_MOVE(path)), m_throws_if_does_not_exist(throws_if_does_not_exist)
		{
		}

	public:
		void execute() const OVERRIDE {
			try {
				real_rmdir(m_path, m_throws_if_does_not_exist);
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

void FilesystemDaemon::load(StreamBuffer &data, const std::string &path, bool throws_if_does_not_exist){
	PROFILE_ME;

	real_load(data, path, throws_if_does_not_exist);
}
void FilesystemDaemon::save(StreamBuffer data, const std::string &path, bool appends, bool throws_if_exists){
	PROFILE_ME;

	real_save(STD_MOVE(data), path, appends, throws_if_exists);
}
void FilesystemDaemon::remove(const std::string &path, bool throws_if_does_not_exist){
	PROFILE_ME;

	real_remove(path, throws_if_does_not_exist);
}
void FilesystemDaemon::rename(const std::string &path, const std::string &new_path){
	PROFILE_ME;

	real_rename(path, new_path);
}
void FilesystemDaemon::mkdir(const std::string &path, bool throws_if_exists){
	PROFILE_ME;

	real_mkdir(path, throws_if_exists);
}
void FilesystemDaemon::rmdir(const std::string &path, bool throws_if_does_not_exist){
	PROFILE_ME;

	real_rmdir(path, throws_if_does_not_exist);
}

boost::shared_ptr<const JobPromise> FilesystemDaemon::enqueue_for_loading(
	boost::shared_ptr<StreamBuffer> data, std::string path, bool throws_if_does_not_exist)
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
	StreamBuffer data, std::string path, bool appends, bool throws_if_exists)
{
	PROFILE_ME;

	AUTO(promise, boost::make_shared<JobPromise>());
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_operations.push_back(boost::make_shared<SaveOperation>(
			promise, STD_MOVE(data), STD_MOVE(path), appends, throws_if_exists));
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
boost::shared_ptr<const JobPromise> FilesystemDaemon::enqueue_for_renaming(
	std::string path, std::string new_path)
{
	PROFILE_ME;

	AUTO(promise, boost::make_shared<JobPromise>());
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_operations.push_back(boost::make_shared<RenameOperation>(
			promise, STD_MOVE(path), STD_MOVE(new_path)));
		g_new_operation.signal();
	}
	return promise;
}
boost::shared_ptr<const JobPromise> FilesystemDaemon::enqueue_for_mkdir(
	std::string path, bool throws_if_exists)
{
	PROFILE_ME;

	AUTO(promise, boost::make_shared<JobPromise>());
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_operations.push_back(boost::make_shared<MkdirOperation>(
			promise, STD_MOVE(path), throws_if_exists));
		g_new_operation.signal();
	}
	return promise;
}
boost::shared_ptr<const JobPromise> FilesystemDaemon::enqueue_for_rmdir(
	std::string path, bool throws_if_does_not_exist)
{
	PROFILE_ME;

	AUTO(promise, boost::make_shared<JobPromise>());
	{
		const Mutex::UniqueLock lock(g_mutex);
		g_operations.push_back(boost::make_shared<RmdirOperation>(
			promise, STD_MOVE(path), throws_if_does_not_exist));
		g_new_operation.signal();
	}
	return promise;
}

}
