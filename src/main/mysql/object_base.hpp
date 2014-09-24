#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#define POSEIDON_MYSQL_OBJECT_BASE_HPP_

#include "../virtual_shared_from_this.hpp"
#include <string>
#include <boost/thread/shared_mutex.hpp>
#include "../atomic.hpp"
#include "../singletons/mysql_daemon.hpp"	// MySqlAsyncLoadCallback

namespace sql {

class Connection;

}

namespace Poseidon {

class MySqlObjectBase
	: public virtual VirtualSharedFromThis
{
private:
	mutable volatile bool m_autoSaves;
	mutable void *volatile m_context;

protected:
	mutable boost::shared_mutex m_mutex;

public:
	MySqlObjectBase()
		: m_autoSaves(false), m_context(0)
	{
	}

protected:
	void invalidate() const;

public:
	void *getContext() const {
		return atomicLoad(m_context);
	}
	void setContext(void *context) const {
		atomicStore(m_context, context);
	}

	bool isAutoSavingEnabled() const {
		return atomicLoad(m_autoSaves);
	}
	void enableAutoSaving() const {
		atomicStore(m_autoSaves, true);
	}
	void disableAutoSaving() const {
		atomicStore(m_autoSaves, false);
	}

	virtual void syncSave(sql::Connection *conn) const = 0;
	virtual bool syncLoad(sql::Connection *conn, const char *filter) = 0;

	void asyncSave() const;
	void asyncLoad(std::string filter,
		MySqlAsyncLoadCallback callback = MySqlAsyncLoadCallback());
};

}

#endif
