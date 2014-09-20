#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#define POSEIDON_MYSQL_OBJECT_BASE_HPP_

#include "../../cxx_ver.hpp"
#include "../virtual_shared_from_this.hpp"
#include <vector>
#include <string>
#include <boost/any.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/function.hpp>

namespace sql {

class Connection;
class PreparedStatement;
class ResultSet;

}

namespace Poseidon {

class MySqlObjectBase
	: public virtual VirtualSharedFromThis
{
private:
	class AsyncPrepareJob;
	class AsyncSaveJob;
	class AsyncLoadJob;

public:
	typedef boost::function<
		void (const boost::shared_ptr<MySqlObjectBase> &)
		> AsyncCallback;

private:
	const char *const m_table;

public:
	explicit MySqlObjectBase(const char *table)
		: m_table(table)
	{
	}

private:
	// 数据库线程处理。
	virtual void prepare(boost::scoped_ptr<sql::PreparedStatement> &ps, sql::Connection *conn) = 0;
	// 主线程处理。
	virtual void pack(sql::PreparedStatement *ps, std::vector<boost::any> &contexts) = 0;

	// 数据库线程处理。
	virtual bool query(boost::scoped_ptr<sql::ResultSet> &rs, sql::Connection *conn,
		const char *filter) const = 0;
	// 主线程处理。
	virtual void fetch(sql::ResultSet *rs) = 0;

public:
	const char *table() const {
		return m_table;
	}

	void asyncLoad(std::string filter, AsyncCallback callback);
	void asyncSave();
};

}

#endif
