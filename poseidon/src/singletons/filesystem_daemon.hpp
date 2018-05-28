// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_FILESYSTEM_DAEMON_HPP_
#define POSEIDON_SINGLETONS_FILESYSTEM_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include <string>
#include "../stream_buffer.hpp"
#include "../promise.hpp"

namespace Poseidon {

struct File_block_read {
	std::uint64_t size_total;
	std::uint64_t begin;
	Stream_buffer data;
};

extern template class Promise_container<File_block_read>;

class File_system_daemon {
public:
	enum {
		limit_eof        = -1ull,
		offset_append    = -2ull,
		offset_truncate  = -3ull,
	};

private:
	File_system_daemon();

public:
	static void start();
	static void stop();

	// 同步接口。
	static File_block_read load(const std::string &path, std::uint64_t begin = 0, std::uint64_t limit = limit_eof, bool throws_if_does_not_exist = true);
	static void save(const std::string &path, Stream_buffer data, std::uint64_t begin = offset_truncate, bool throws_if_exists = false);
	static void remove(const std::string &path, bool throws_if_does_not_exist = true);
	static void rename(const std::string &path, const std::string &new_path);
	static void mkdir(const std::string &path, bool throws_if_exists = false);
	static void rmdir(const std::string &path, bool throws_if_does_not_exist = true);

	// 异步接口。
	static boost::shared_ptr<const Promise_container<File_block_read> > enqueue_for_loading(std::string path, std::uint64_t begin = 0, std::uint64_t limit = limit_eof, bool throws_if_does_not_exist = true);
	static boost::shared_ptr<const Promise> enqueue_for_saving(std::string path, Stream_buffer data, std::uint64_t begin = offset_truncate, bool throws_if_exists = false);
	static boost::shared_ptr<const Promise> enqueue_for_removing(std::string path, bool throws_if_does_not_exist = true);
	static boost::shared_ptr<const Promise> enqueue_for_renaming(std::string path, std::string new_path);
	static boost::shared_ptr<const Promise> enqueue_for_mkdir(std::string path, bool throws_if_exists = false);
	static boost::shared_ptr<const Promise> enqueue_for_rmdir(std::string path, bool throws_if_does_not_exist = true);
};

}

#endif
