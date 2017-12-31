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

struct FileBlockRead {
	boost::uint64_t size_total;
	boost::uint64_t begin;
	StreamBuffer data;
};

extern template class PromiseContainer<FileBlockRead>;

class FileSystemDaemon {
public:
	static CONSTEXPR const boost::uint64_t LIMIT_EOF       = (boost::uint64_t)-1;
	static CONSTEXPR const boost::uint64_t OFFSET_APPEND   = (boost::uint64_t)-2;
	static CONSTEXPR const boost::uint64_t OFFSET_TRUNCATE = (boost::uint64_t)-3;

private:
	FileSystemDaemon();

public:
	static void start();
	static void stop();

	// 同步接口。
	static FileBlockRead load(const std::string &path, boost::uint64_t begin = 0, boost::uint64_t limit = LIMIT_EOF, bool throws_if_does_not_exist = true);
	static void save(const std::string &path, StreamBuffer data, boost::uint64_t begin = OFFSET_TRUNCATE, bool throws_if_exists = false);
	static void remove(const std::string &path, bool throws_if_does_not_exist = true);
	static void rename(const std::string &path, const std::string &new_path);
	static void mkdir(const std::string &path, bool throws_if_exists = false);
	static void rmdir(const std::string &path, bool throws_if_does_not_exist = true);

	// 异步接口。
	static boost::shared_ptr<const PromiseContainer<FileBlockRead> > enqueue_for_loading(std::string path, boost::uint64_t begin = 0, boost::uint64_t limit = LIMIT_EOF, bool throws_if_does_not_exist = true);
	static boost::shared_ptr<const Promise> enqueue_for_saving(std::string path, StreamBuffer data, boost::uint64_t begin = OFFSET_TRUNCATE, bool throws_if_exists = false);
	static boost::shared_ptr<const Promise> enqueue_for_removing(std::string path, bool throws_if_does_not_exist = true);
	static boost::shared_ptr<const Promise> enqueue_for_renaming(std::string path, std::string new_path);
	static boost::shared_ptr<const Promise> enqueue_for_mkdir(std::string path, bool throws_if_exists = false);
	static boost::shared_ptr<const Promise> enqueue_for_rmdir(std::string path, bool throws_if_does_not_exist = true);
};

}

#endif
