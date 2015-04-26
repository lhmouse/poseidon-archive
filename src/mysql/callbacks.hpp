// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_CALLBACKS_HPP_
#define POSEIDON_MYSQL_CALLBACKS_HPP_

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace Poseidon {

namespace MySql {
	class ObjectBase;

	// ExceptionCallback 可能在主线程或数据库线程中被调用。
	typedef boost::function<
		void ()
		> ExceptionCallback;

	// 这三个回调一定是在主线程中调用。
	typedef boost::function<
		void (bool succeeded, unsigned long long autoIncrementId)
		> AsyncSaveCallback;

	typedef boost::function<
		void (bool found)
		> AsyncLoadCallback;

	typedef boost::function<
		void (std::vector<boost::shared_ptr<ObjectBase> > objects)
		> BatchAsyncLoadCallback;
}

}

#endif
