// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MONGODB_OBJECT_BASE_HPP_
#define POSEIDON_MONGODB_OBJECT_BASE_HPP_

#include "../cxx_ver.hpp"
#include "../cxx_util.hpp"
#include "connection.hpp"
#include "bson_builder.hpp"
#include "exception.hpp"
#include <string>
#include <exception>
#include <iosfwd>
#include <cstdio>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/cstdint.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>
#include "../atomic.hpp"
#include "../shared_nts.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../recursive_mutex.hpp"
#include "../virtual_shared_from_this.hpp"
#include "../uuid.hpp"

namespace Poseidon {
namespace MongoDb {

class ObjectBase : NONCOPYABLE, public virtual VirtualSharedFromThis {
public:
	template<typename ValueT> class Field;

private:
	mutable volatile bool m_auto_saves;
	mutable void *volatile m_combined_write_stamp;

protected:
	mutable RecursiveMutex m_mutex;

public:
	ObjectBase()
		: m_auto_saves(false), m_combined_write_stamp(NULLPTR)
	{ }
	// 不要不写析构函数，否则 RTTI 将无法在动态库中使用。
	~ObjectBase();

public:
	bool is_auto_saving_enabled() const NOEXCEPT;
	void enable_auto_saving() const NOEXCEPT;
	void disable_auto_saving() const NOEXCEPT;

	bool invalidate() const NOEXCEPT;

	void *get_combined_write_stamp() const NOEXCEPT;
	void set_combined_write_stamp(void *stamp) const NOEXCEPT;

	virtual const char *get_collection() const = 0;
	virtual void generate_document(BsonBuilder &doc) const = 0;
	virtual std::string generate_primary_key() const = 0;
	virtual void fetch(const boost::shared_ptr<const Connection> &conn) = 0;
};

template<typename ValueT>
class ObjectBase::Field : NONCOPYABLE {
private:
	ObjectBase *const m_parent;
	ValueT m_value;

public:
	explicit Field(ObjectBase *parent, ValueT value = ValueT())
		: m_parent(parent), m_value(STD_MOVE_IDN(value))
	{ }

public:
	const ValueT &unlocked_get() const {
		return m_value;
	}
	ValueT get() const {
		const RecursiveMutex::UniqueLock lock(m_parent->m_mutex);
		return m_value;
	}
	void set(ValueT value, bool invalidates_parent = true){
		const RecursiveMutex::UniqueLock lock(m_parent->m_mutex);
		m_value = STD_MOVE_IDN(value);

		if(invalidates_parent){
			m_parent->invalidate();
		}
	}

public:
	operator const ValueT &() const {
		return unlocked_get();
	}
	Field &operator=(ValueT value){
		set(STD_MOVE_IDN(value));
		return *this;
	}
};

extern template class ObjectBase::Field<bool>;
extern template class ObjectBase::Field<boost::int64_t>;
extern template class ObjectBase::Field<boost::uint64_t>;
extern template class ObjectBase::Field<double>;
extern template class ObjectBase::Field<std::string>;
extern template class ObjectBase::Field<Uuid>;
extern template class ObjectBase::Field<std::basic_string<unsigned char> >;

template<typename ValueT>
inline std::ostream &operator<<(std::ostream &os, const ObjectBase::Field<ValueT> &rhs){
	return os <<rhs.unlocked_get();
}
template<typename ValueT>
inline std::istream &operator>>(std::istream &is, ObjectBase::Field<ValueT> &rhs){
	ValueT value;
	if(is >>value){
		rhs.set(STD_MOVE(value));
	}
	return is;
}

extern void enqueue_for_saving(const boost::shared_ptr<ObjectBase> &obj);

template<typename ObjectT>
typename boost::enable_if_c<boost::is_base_of<ObjectBase, ObjectT>::value,
	const boost::shared_ptr<ObjectT> &>::type begin_synchronization(const boost::shared_ptr<ObjectT> &obj, bool save_now)
{
	obj->enable_auto_saving();
	if(save_now){
		enqueue_for_saving(obj);
	}
	return obj;
}

}
}

#endif
