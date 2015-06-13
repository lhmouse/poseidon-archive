// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#define POSEIDON_MYSQL_OBJECT_BASE_HPP_

#include "../cxx_ver.hpp"
#include "../cxx_util.hpp"
#include "callbacks.hpp"
#include "connection.hpp"
#include "utilities.hpp"
#include <string>
#include <sstream>
#include <cstdio>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/cstdint.hpp>
#include "../atomic.hpp"
#include "../log.hpp"
#include "../mutex.hpp"
#include "../virtual_shared_from_this.hpp"

namespace Poseidon {

namespace MySql {
	// 注意 ExceptionCallback 不是线程安全的。

	class ObjectBase : NONCOPYABLE, public virtual VirtualSharedFromThis {
	protected:
		static void batchLoad(boost::shared_ptr<ObjectBase> (*factory)(),
			const char *tableHint, std::string query,
			BatchAsyncLoadCallback callback, ExceptionCallback except);

	private:
		mutable volatile bool m_autoSaves;

	protected:
		mutable Mutex m_mutex;

	protected:
		ObjectBase();
		// 不要不写析构函数，否则 RTTI 将无法在动态库中使用。
		~ObjectBase();

	protected:
		bool invalidate() const NOEXCEPT;

	public:
		bool isAutoSavingEnabled() const {
			return atomicLoad(m_autoSaves, ATOMIC_CONSUME);
		}
		void enableAutoSaving() const {
			atomicStore(m_autoSaves, true, ATOMIC_RELEASE);
		}
		void disableAutoSaving() const {
			atomicStore(m_autoSaves, false, ATOMIC_RELEASE);
		}

		virtual const char *getTableName() const = 0;

		virtual void syncGenerateSql(std::string &sql, bool toReplace) const = 0;
		virtual void syncFetch(const Connection &conn) = 0;

		void asyncSave(bool toReplace, bool urgent = false, AsyncSaveCallback callback = AsyncSaveCallback(),
			ExceptionCallback except = ExceptionCallback()) const;
		void asyncLoad(std::string query, AsyncLoadCallback callback,
			ExceptionCallback except = ExceptionCallback());
	};
}

}

#endif
