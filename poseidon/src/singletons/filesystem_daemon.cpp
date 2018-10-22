// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

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
#include "../promise.hpp"
#include "../profiler.hpp"

namespace Poseidon {

template class Promise_container<File_block_read>;

namespace {
	File_block_read real_load(const std::string &path, boost::uint64_t begin, boost::uint64_t limit, bool throws_if_does_not_exist){
		File_block_read block = { };

		int flags = O_RDONLY;
		Unique_file file;
		if(!file.reset(::open(path.c_str(), flags))){
			const int err_code = errno;
			if(!throws_if_does_not_exist && (err_code == ENOENT)){
				return block;
			}
			POSEIDON_LOG_ERROR("Failed to load file: path = ", path, ", err_code = ", err_code);
			POSEIDON_THROW(System_exception, err_code);
		}
		struct ::stat stat_buf;
		if(::fstat(file.get(), &stat_buf) != 0){
			const int err_code = errno;
			POSEIDON_LOG_ERROR("Failed to retrieve file information: path = ", path, ", err_code = ", err_code);
			POSEIDON_THROW(System_exception, err_code);
		}
		if(begin != 0){
			if(::lseek(file.get(), static_cast< ::off_t>(begin), SEEK_SET) == (::off_t)-1){
				const int err_code = errno;
				POSEIDON_LOG_ERROR("Failed to seek file: path = ", path, ", err_code = ", err_code);
				POSEIDON_THROW(System_exception, err_code);
			}
		}

		block.size_total = static_cast<boost::uint64_t>(stat_buf.st_size);
		block.begin = begin;

		std::size_t bytes_read = 0;
		for(;;){
			char temp[16384];
			std::size_t avail;
			if(limit == Filesystem_daemon::limit_eof){
				avail = sizeof(temp);
			} else {
				avail = static_cast<std::size_t>(std::min<boost::uint64_t>(limit - bytes_read, sizeof(temp)));
			}
			if(avail == 0){
				break;
			}
			const ::ssize_t result = ::read(file.get(), temp, avail);
			if(result == 0){
				break;
			}
			if(result < 0){
				const int err_code = errno;
				POSEIDON_LOG_ERROR("Error loading file: path = ", path, ", err_code = ", err_code);
				POSEIDON_THROW(System_exception, err_code);
			}
			avail = static_cast<std::size_t>(result);
			block.data.put(temp, avail);
			bytes_read += avail;
		}
		POSEIDON_LOG_DEBUG("Finished loading file: path = ", path, ", bytes_read = ", bytes_read);
		return block;
	}
	void real_save(const std::string &path, Stream_buffer data, boost::uint64_t begin, bool throws_if_exists){
		int flags = O_CREAT | O_WRONLY;
		if(begin == Filesystem_daemon::offset_append){
			flags |= O_APPEND;
		} else if(begin == Filesystem_daemon::offset_truncate){
			flags |= O_TRUNC;
		}
		if(throws_if_exists){
			flags |= O_EXCL;
		}
		Unique_file file;
		if(!file.reset(::open(path.c_str(), flags, static_cast< ::mode_t>(0666)))){
			const int err_code = errno;
			POSEIDON_LOG_ERROR("Failed to save file: path = ", path, ", err_code = ", err_code);
			POSEIDON_THROW(System_exception, err_code);
		}
		if(!(flags & (O_APPEND | O_TRUNC)) && (begin != 0)){
			if(::lseek(file.get(), static_cast< ::off_t>(begin), SEEK_SET) == (::off_t)-1){
				const int err_code = errno;
				POSEIDON_LOG_ERROR("Failed to seek file: path = ", path, ", err_code = ", err_code);
				POSEIDON_THROW(System_exception, err_code);
			}
		}

		std::size_t bytes_written = 0;
		for(;;){
			char temp[16384];
			std::size_t avail = data.get(temp, sizeof(temp));
			if(avail == 0){
				break;
			}
			::ssize_t result = ::write(file.get(), temp, avail);
			if(result < 0){
				const int err_code = errno;
				POSEIDON_LOG_ERROR("Error saving file: path = ", path, ", err_code = ", err_code);
				POSEIDON_THROW(System_exception, err_code);
			}
			bytes_written += avail;
		}
		POSEIDON_LOG_DEBUG("Finished saving file: path = ", path, ", bytes_written = ", bytes_written);
	}
	void real_remove(const std::string &path, bool throws_if_does_not_exist){
		if(::unlink(path.c_str()) != 0){
			const int err_code = errno;
			if(!throws_if_does_not_exist && (err_code == ENOENT)){
				return;
			}
			POSEIDON_LOG_ERROR("Failed to remove file: path = ", path, ", err_code = ", err_code);
			POSEIDON_THROW(System_exception, err_code);
		}
	}
	void real_rename(const std::string &path, const std::string &new_path){
		if(::rename(path.c_str(), new_path.c_str()) != 0){
			const int err_code = errno;
			POSEIDON_LOG_ERROR("Failed to rename file: path = ", path, ", err_code = ", err_code);
			POSEIDON_THROW(System_exception, err_code);
		}
	}
	void real_mkdir(const std::string &path, bool throws_if_exists){
		if(::mkdir(path.c_str(), static_cast< ::mode_t>(0777)) != 0){
			const int err_code = errno;
			if(!throws_if_exists && (err_code == EEXIST)){
				struct ::stat stat_buf;
				if((::stat(path.c_str(), &stat_buf) == 0) && S_ISDIR(stat_buf.st_mode)){
					return;
				}
			}
			POSEIDON_LOG_ERROR("Failed to make directory: path = ", path, ", err_code = ", err_code);
			POSEIDON_THROW(System_exception, err_code);
		}
	}
	void real_rmdir(const std::string &path, bool throws_if_does_not_exist){
		if(::rmdir(path.c_str()) != 0){
			const int err_code = errno;
			if(!throws_if_does_not_exist && (err_code == ENOENT)){
				return;
			}
			POSEIDON_LOG_ERROR("Failed to remove directory: path = ", path, ", err_code = ", err_code);
			POSEIDON_THROW(System_exception, err_code);
		}
	}

	class Operation_base : NONCOPYABLE {
	private:
		const boost::weak_ptr<Promise> m_weak_promise;
		const std::string m_path;

	public:
		Operation_base(const boost::shared_ptr<Promise> &promise, std::string path)
			: m_weak_promise(promise), m_path(STD_MOVE(path))
		{
			//
		}
		virtual ~Operation_base(){
			//
		}

	public:
		const std::string & get_path() const {
			return m_path;
		}

		virtual boost::shared_ptr<Promise> get_promise() const {
			return m_weak_promise.lock();
		}
		virtual void execute() = 0;
	};

	class Load_operation : public Operation_base {
	private:
		boost::shared_ptr<Promise_container<File_block_read> > m_promised_block;
		boost::uint64_t m_begin;
		boost::uint64_t m_limit;
		bool m_throws_if_does_not_exist;

	public:
		Load_operation(boost::shared_ptr<Promise_container<File_block_read> > promised_block, std::string path, boost::uint64_t begin, boost::uint64_t limit, bool throws_if_does_not_exist)
			: Operation_base(promised_block, STD_MOVE(path))
			, m_promised_block(STD_MOVE(promised_block)), m_begin(begin), m_limit(limit), m_throws_if_does_not_exist(throws_if_does_not_exist)
		{
			//
		}

	public:
		void execute() OVERRIDE {
			POSEIDON_PROFILE_ME;

			if(!get_promise()){
				POSEIDON_LOG_DEBUG("Discarding isolated loading operation: path = ", get_path());
				return;
			}
			AUTO(block, real_load(get_path(), m_begin, m_limit, m_throws_if_does_not_exist));
			m_promised_block->set_success(STD_MOVE(block));
		}
	};

	class Save_operation : public Operation_base {
	private:
		Stream_buffer m_data;
		boost::uint64_t m_begin;
		bool m_throws_if_exists;

	public:
		Save_operation(const boost::shared_ptr<Promise> &promise, std::string path, Stream_buffer data, boost::uint64_t begin, bool throws_if_exists)
			: Operation_base(promise, STD_MOVE(path))
			, m_data(STD_MOVE(data)), m_begin(begin), m_throws_if_exists(throws_if_exists)
		{
			//
		}

	public:
		void execute() OVERRIDE {
			POSEIDON_PROFILE_ME;

			real_save(get_path(), m_data, m_begin, m_throws_if_exists);
		}
	};

	class Remove_operation : public Operation_base {
	private:
		bool m_throws_if_does_not_exist;

	public:
		Remove_operation(const boost::shared_ptr<Promise> &promise, std::string path, bool throws_if_does_not_exist)
			: Operation_base(promise, STD_MOVE(path))
			, m_throws_if_does_not_exist(throws_if_does_not_exist)
		{
			//
		}

	public:
		void execute() OVERRIDE {
			POSEIDON_PROFILE_ME;

			real_remove(get_path(), m_throws_if_does_not_exist);
		}
	};

	class Rename_operation : public Operation_base {
	private:
		std::string m_new_path;

	public:
		Rename_operation(const boost::shared_ptr<Promise> &promise, std::string path, std::string new_path)
			: Operation_base(promise, STD_MOVE(path))
			, m_new_path(STD_MOVE(new_path))
		{
			//
		}

	public:
		void execute() OVERRIDE {
			POSEIDON_PROFILE_ME;

			real_rename(get_path(), m_new_path);
		}
	};

	class Mkdir_operation : public Operation_base {
	private:
		bool m_throws_if_exists;

	public:
		Mkdir_operation(const boost::shared_ptr<Promise> &promise, std::string path, bool throws_if_exists)
			: Operation_base(promise, STD_MOVE(path))
			, m_throws_if_exists(throws_if_exists)
		{
			//
		}

	public:
		void execute() OVERRIDE {
			POSEIDON_PROFILE_ME;

			real_mkdir(get_path(), m_throws_if_exists);
		}
	};

	class Rmdir_operation : public Operation_base {
	private:
		bool m_throws_if_does_not_exist;

	public:
		Rmdir_operation(const boost::shared_ptr<Promise> &promise, std::string path, bool throws_if_does_not_exist)
			: Operation_base(promise, STD_MOVE(path))
			, m_throws_if_does_not_exist(throws_if_does_not_exist)
		{
			//
		}

	public:
		void execute() OVERRIDE {
			POSEIDON_PROFILE_ME;

			real_rmdir(get_path(), m_throws_if_does_not_exist);
		}
	};

	volatile bool g_running = false;
	Thread g_thread;

	struct Operation_queue_element {
		boost::shared_ptr<Operation_base> operation;
	};

	Mutex g_mutex;
	Condition_variable g_new_operation;
	boost::container::deque<Operation_queue_element> g_operations;

	bool pump_one_element() NOEXCEPT {
		POSEIDON_PROFILE_ME;

		Operation_queue_element *elem;
		{
			const Mutex::Unique_lock lock(g_mutex);
			if(g_operations.empty()){
				return false;
			}
			elem = &g_operations.front();
		}
		STD_EXCEPTION_PTR except;
		try {
			elem->operation->execute();
		} catch(std::exception &e){
			POSEIDON_LOG_WARNING("std::exception thrown: what = ", e.what());
			except = STD_CURRENT_EXCEPTION();
		} catch(...){
			POSEIDON_LOG_WARNING("Unknown exception thrown.");
			except = STD_CURRENT_EXCEPTION();
		}
		const AUTO(promise, elem->operation->get_promise());
		if(promise){
			if(except){
				promise->set_exception(STD_MOVE(except), false);
			} else {
				promise->set_success(false);
			}
		}
		const Mutex::Unique_lock lock(g_mutex);
		g_operations.pop_front();
		return true;
	}

	void thread_proc(){
		POSEIDON_PROFILE_ME;
		POSEIDON_LOG(Logger::special_major | Logger::level_info, "Filesystem daemon started.");

		unsigned timeout = 0;
		for(;;){
			bool busy;
			do {
				busy = pump_one_element();
				timeout = std::min(timeout * 2u + 1u, !busy * 100u);
			} while(busy);

			Mutex::Unique_lock lock(g_mutex);
			if(!atomic_load(g_running, memory_order_consume)){
				break;
			}
			g_new_operation.timed_wait(lock, timeout);
		}

		POSEIDON_LOG(Logger::special_major | Logger::level_info, "Filesystem daemon stopped.");
	}

	void submit_operation(boost::shared_ptr<Operation_base> operation){
		POSEIDON_PROFILE_ME;

		const Mutex::Unique_lock lock(g_mutex);
		Operation_queue_element elem = { STD_MOVE(operation) };
		g_operations.push_back(STD_MOVE(elem));
		g_new_operation.signal();
	}
}

void Filesystem_daemon::start(){
	if(atomic_exchange(g_running, true, memory_order_acq_rel) != false){
		POSEIDON_LOG_FATAL("Only one daemon is allowed at the same time.");
		std::terminate();
	}
	POSEIDON_LOG(Logger::special_major | Logger::level_info, "Starting Filesystem daemon...");

	Thread(&thread_proc, Rcnts::view(" F  "), Rcnts::view("Filesystem")).swap(g_thread);
}
void Filesystem_daemon::stop(){
	if(atomic_exchange(g_running, false, memory_order_acq_rel) == false){
		return;
	}
	POSEIDON_LOG(Logger::special_major | Logger::level_info, "Stopping Filesystem daemon...");

	if(g_thread.joinable()){
		g_thread.join();
	}

	const Mutex::Unique_lock lock(g_mutex);
	g_operations.clear();
}

File_block_read Filesystem_daemon::load(const std::string &path, boost::uint64_t begin, boost::uint64_t limit, bool throws_if_does_not_exist){
	POSEIDON_PROFILE_ME;

	return real_load(path, begin, limit, throws_if_does_not_exist);
}
void Filesystem_daemon::save(const std::string &path, Stream_buffer data, boost::uint64_t begin, bool throws_if_exists){
	POSEIDON_PROFILE_ME;

	real_save(path, STD_MOVE(data), begin, throws_if_exists);
}
void Filesystem_daemon::remove(const std::string &path, bool throws_if_does_not_exist){
	POSEIDON_PROFILE_ME;

	real_remove(path, throws_if_does_not_exist);
}
void Filesystem_daemon::rename(const std::string &path, const std::string &new_path){
	POSEIDON_PROFILE_ME;

	real_rename(path, new_path);
}
void Filesystem_daemon::mkdir(const std::string &path, bool throws_if_exists){
	POSEIDON_PROFILE_ME;

	real_mkdir(path, throws_if_exists);
}
void Filesystem_daemon::rmdir(const std::string &path, bool throws_if_does_not_exist){
	POSEIDON_PROFILE_ME;

	real_rmdir(path, throws_if_does_not_exist);
}

boost::shared_ptr<const Promise_container<File_block_read> > Filesystem_daemon::enqueue_for_loading(std::string path, boost::uint64_t begin, boost::uint64_t limit, bool throws_if_does_not_exist){
	POSEIDON_PROFILE_ME;

	AUTO(promise, boost::make_shared<Promise_container<File_block_read> >());
	AUTO(operation, boost::make_shared<Load_operation>(promise, STD_MOVE(path), begin, limit, throws_if_does_not_exist));
	submit_operation(STD_MOVE_IDN(operation));
	return promise;
}
boost::shared_ptr<const Promise> Filesystem_daemon::enqueue_for_saving(std::string path, Stream_buffer data, boost::uint64_t begin, bool throws_if_exists){
	POSEIDON_PROFILE_ME;

	AUTO(promise, boost::make_shared<Promise>());
	AUTO(operation, boost::make_shared<Save_operation>(promise, STD_MOVE(path), STD_MOVE(data), begin, throws_if_exists));
	submit_operation(STD_MOVE_IDN(operation));
	return promise;
}
boost::shared_ptr<const Promise> Filesystem_daemon::enqueue_for_removing(std::string path, bool throws_if_does_not_exist){
	POSEIDON_PROFILE_ME;

	AUTO(promise, boost::make_shared<Promise>());
	AUTO(operation, boost::make_shared<Remove_operation>(promise, STD_MOVE(path), throws_if_does_not_exist));
	submit_operation(STD_MOVE_IDN(operation));
	return promise;
}
boost::shared_ptr<const Promise> Filesystem_daemon::enqueue_for_renaming(std::string path, std::string new_path){
	POSEIDON_PROFILE_ME;

	AUTO(promise, boost::make_shared<Promise>());
	AUTO(operation, boost::make_shared<Rename_operation>(promise, STD_MOVE(path), STD_MOVE(new_path)));
	submit_operation(STD_MOVE_IDN(operation));
	return promise;
}
boost::shared_ptr<const Promise> Filesystem_daemon::enqueue_for_mkdir(std::string path, bool throws_if_exists){
	POSEIDON_PROFILE_ME;

	AUTO(promise, boost::make_shared<Promise>());
	AUTO(operation, boost::make_shared<Mkdir_operation>(promise, STD_MOVE(path), throws_if_exists));
	submit_operation(STD_MOVE_IDN(operation));
	return promise;
}
boost::shared_ptr<const Promise> Filesystem_daemon::enqueue_for_rmdir(std::string path, bool throws_if_does_not_exist){
	POSEIDON_PROFILE_ME;

	AUTO(promise, boost::make_shared<Promise>());
	AUTO(operation, boost::make_shared<Rmdir_operation>(promise, STD_MOVE(path), throws_if_does_not_exist));
	submit_operation(STD_MOVE_IDN(operation));
	return promise;
}

}
