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
		static void batch_load(std::vector<boost::shared_ptr<ObjectBase> > &ret,
			boost::shared_ptr<ObjectBase> (*factory)(), const char *table_hint, std::string query);

	private:
		mutable volatile bool m_auto_saves;
		mutable void *volatile m_combined_write_stamp;

	protected:
		mutable Mutex m_mutex;

	protected:
		ObjectBase();
		// 不要不写析构函数，否则 RTTI 将无法在动态库中使用。
		~ObjectBase();

	protected:
		bool invalidate() const NOEXCEPT;

	public:
		bool is_auto_saving_enabled() const;
		void enable_auto_saving() const;
		void disable_auto_saving() const;

		void *get_combined_write_stamp() const;
		void set_combined_write_stamp(void *stamp) const;

		virtual const char *get_table_name() const = 0;

		virtual void generate_sql(std::string &sql, bool to_replace) const = 0;
		virtual void fetch(const boost::shared_ptr<const Connection> &conn) = 0;

		void save_and_wait(bool to_replace) const;
		void load_and_wait(std::string query);
		void async_save(bool to_replace, bool urgent = false) const;
	};
}

}

#endif
