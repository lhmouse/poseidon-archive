// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MONGODB_OBJECT_BASE_HPP_
#define POSEIDON_MONGODB_OBJECT_BASE_HPP_

#include "../cxx_ver.hpp"
#include "../cxx_util.hpp"
#include "connection.hpp"
#include "oid.hpp"
#include "bson_builder.hpp"
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

namespace MongoDb {
	class ObjectBase : NONCOPYABLE, public virtual VirtualSharedFromThis {
	private:
		mutable volatile bool m_auto_saves;
		mutable void *volatile m_combined_write_stamp;

		Oid m_oid;

	protected:
		mutable Mutex m_mutex;

	protected:
		ObjectBase();
		// 不要不写析构函数，否则 RTTI 将无法在动态库中使用。
		~ObjectBase();

	public:
		bool is_auto_saving_enabled() const;
		void enable_auto_saving() const;
		void disable_auto_saving() const;

		bool invalidate() const NOEXCEPT;

		void *get_combined_write_stamp() const;
		void set_combined_write_stamp(void *stamp) const;

		virtual const char *get_collection_name() const = 0;

		virtual void generate_document(BsonBuilder &doc) const = 0;
		virtual void generate_primary_key(BsonBuilder &query) const = 0;
		virtual void fetch(const boost::shared_ptr<const Connection> &conn) = 0;
		void async_save(bool to_replace, bool urgent = false) const;

		const Oid &unlocked_get_oid() const {
			return m_oid;
		}
		Oid get_oid() const {
			const Mutex::UniqueLock lock(m_mutex);
			return m_oid;
		}
		void set_oid(const Oid &val, bool invalidates = true){
			{
				const Mutex::UniqueLock lock(m_mutex);
				m_oid = val;
			}
			if(invalidates){
				invalidate();
			}
		}
	};
}

}

#endif
