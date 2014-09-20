#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#define POSEIDON_MYSQL_OBJECT_BASE_HPP_

#include "../../cxx_ver.hpp"
#include "../virtual_shared_from_this.hpp"
#include <vector>
#include <string>
#include <boost/thread/mutex.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace sql {

class Connection;
class PreparedStatement;
class ResultSet;

}

namespace Poseidon {

class MySqlFieldBase;

class MySqlObjectBase
	: public virtual VirtualSharedFromThis
{
	friend class MySqlFieldBase;

private:
	const char *const m_table;

	mutable boost::mutex m_fieldMutex;
	std::vector<boost::reference_wrapper<MySqlFieldBase> > m_fields;

public:
	explicit MySqlObjectBase(const char *table);

public:
	void syncLoad(sql::Connection *conn, const std::string &filter);
	void syncSave(sql::Connection *conn);

	void asyncLoad(std::string filter,
		boost::function<void (const boost::shared_ptr<MySqlObjectBase> &)> callback);
	void asyncSave();
};

}

#endif
