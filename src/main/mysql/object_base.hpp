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

namespace Poseidon {

class MySqlObjectBase
	: public virtual VirtualSharedFromThis
{
private:
	const char *const m_table;

	unsigned long long m_timeStamp;
	mutable boost::mutex m_fieldMutex;
	std::vector<boost::reference_wrapper<class MySqlFieldBase> > m_fields;

public:
	explicit MySqlObjectBase(const char *table);

public:
	// 参数是 SQL 里面的 where 子句。
	void asyncLoad(const std::string &where,
		boost::function<void (const boost::shared_ptr<MySqlObjectBase> &)> callback);

	// 可能返回空字符串。
	std::string dumpSql(bool invalidatedOnly = true);
	void notifyStored(unsigned long long time);
};

}

#endif
