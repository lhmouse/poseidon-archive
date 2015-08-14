// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#define POSEIDON_MYSQL_OBJECT_BASE_HPP_

#include "../cxx_ver.hpp"
#include "../cxx_util.hpp"
#include "connection.hpp"
#include "utilities.hpp"
#include "exception.hpp"
#include <string>
#include <sstream>
#include <vector>
#include <exception>
#include <cstdio>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/function.hpp>
#include <boost/cstdint.hpp>
#include "../atomic.hpp"
#include "../shared_nts.hpp"
#include "../log.hpp"
#include "../mutex.hpp"
#include "../virtual_shared_from_this.hpp"
#include "../uuid.hpp"

namespace Poseidon {

namespace MySql {
	class ObjectBase : NONCOPYABLE, public virtual VirtualSharedFromThis {
	protected:
		static void batchLoad(std::vector<boost::shared_ptr<ObjectBase> > &ret,
			boost::shared_ptr<ObjectBase> (*factory)(), const char *tableHint, std::string query);

	private:
		mutable volatile bool m_autoSaves;
		mutable void *volatile m_combinedWriteStamp;

	protected:
		mutable Mutex m_mutex;

	protected:
		ObjectBase();
		// 不要不写析构函数，否则 RTTI 将无法在动态库中使用。
		~ObjectBase();

	protected:
		bool invalidate() const NOEXCEPT;

	public:
		bool isAutoSavingEnabled() const;
		void enableAutoSaving() const;
		void disableAutoSaving() const;

		void *getCombinedWriteStamp() const;
		void setCombinedWriteStamp(void *stamp) const;

		virtual const char *getTableName() const = 0;

		virtual void syncGenerateSql(std::string &sql, bool toReplace) const = 0;
		virtual void syncFetch(const boost::shared_ptr<const Connection> &conn) = 0;

		void syncSave(bool toReplace) const;
		void syncLoad(std::string query);
		void asyncSave(bool toReplace, bool urgent = false) const;
	};
}

}

#endif
