// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_PROMISE_HPP_
#define POSEIDON_MYSQL_PROMISE_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/exception_ptr.hpp>
#include "../mutex.hpp"

namespace Poseidon {

namespace MySql {
	class Exception;

	class Promise : NONCOPYABLE {
	private:
		mutable Mutex m_mutex;
		bool m_satisfied;
		boost::exception_ptr m_except;

	public:
		Promise();
		~Promise();

	public:
		bool isSatisfied() const;
		void checkAndRethrow() const;

		void setSuccess();
		void setException(const boost::exception_ptr &except);
	};
}

}

#endif
