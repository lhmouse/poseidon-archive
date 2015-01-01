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
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/cstdint.hpp>
#include "../atomic.hpp"
#include "../log.hpp"
#include "../virtual_shared_from_this.hpp"
#include "../utilities.hpp"

namespace Poseidon {

class MySqlObjectBase : NONCOPYABLE
	, public virtual VirtualSharedFromThis
{
protected:
	static void batchAsyncLoad(const char *tableHint, std::string query,
		boost::shared_ptr<MySqlObjectBase> (*factory)(),
		MySqlBatchAsyncLoadCallback callback, MySqlExceptionCallback except);

private:
	mutable volatile bool m_autoSaves;
	mutable const void *m_context;

protected:
	mutable boost::shared_mutex m_mutex;

protected:
	MySqlObjectBase();
	// 不要不写析构函数，否则 RTTI 将无法在动态库中使用。
	~MySqlObjectBase();

protected:
	void invalidate() const;

public:
	bool isAutoSavingEnabled() const {
		return atomicLoad(m_autoSaves, ATOMIC_ACQUIRE);
	}
	void enableAutoSaving() const {
		atomicStore(m_autoSaves, true, ATOMIC_RELEASE);
	}
	void disableAutoSaving() const {
		atomicStore(m_autoSaves, false, ATOMIC_RELEASE);
	}

	// 用于写入合并时标记最后一次队列节点的地址。
	const void *getContext() const {
		return atomicLoad(m_context, ATOMIC_ACQUIRE);
	}
	void setContext(const void *context) const {
		atomicStore(m_context, context, ATOMIC_RELEASE);
	}

	virtual const char *getTableName() const = 0;

	virtual void syncGenerateSql(std::string &sql, bool replaces) const = 0;
	virtual void syncFetch(const MySqlConnection &conn) = 0;

	void asyncSave(bool replaces, MySqlAsyncSaveCallback callback = MySqlAsyncSaveCallback(),
		MySqlExceptionCallback except = MySqlExceptionCallback()) const;
	void asyncLoad(std::string query, MySqlAsyncLoadCallback callback,
		MySqlExceptionCallback except = MySqlExceptionCallback());
};

}

#endif
