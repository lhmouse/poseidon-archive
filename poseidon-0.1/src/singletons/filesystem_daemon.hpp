// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_FILESYSTEM_DAEMON_HPP_
#define POSEIDON_SINGLETONS_FILESYSTEM_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <string>

namespace Poseidon {

class JobPromise;

class FilesystemDaemon {
private:
	FilesystemDaemon();

public:
	static void start();
	static void stop();

	// 同步接口。
	static void load(std::string &data, const std::string &path, bool throws_if_does_not_exist = true);
	static void save(std::string data, const std::string &path, bool appends, bool forces_creation);
	static void remove(const std::string &path, bool throws_if_does_not_exist = true);

	// 异步接口。
	// 以下第一个参数是出参。
	static boost::shared_ptr<const JobPromise> enqueue_for_loading(
		boost::shared_ptr<std::string> data, std::string path, bool throws_if_does_not_exist = true);
	static boost::shared_ptr<const JobPromise> enqueue_for_saving(
		std::string data, std::string path, bool appends, bool forces_creation);
	static boost::shared_ptr<const JobPromise> enqueue_for_removing(
		std::string path, bool throws_if_does_not_exist = true);
};

}

#endif
