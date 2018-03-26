// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#define POSEIDON_MYSQL_OBJECT_BASE_HPP_

#include "../cxx_ver.hpp"
#include "../cxx_util.hpp"
#include "connection.hpp"
#include "formatting.hpp"
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
#include "../shared_nts.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../recursive_mutex.hpp"
#include "../virtual_shared_from_this.hpp"
#include "../uuid.hpp"

namespace Poseidon {
namespace Mysql {

class Object_base : NONCOPYABLE, public virtual Virtual_shared_from_this {
public:
	template<typename ValueT> class Field;
	class Delimiter;

private:
	mutable volatile bool m_auto_saves;
	mutable void *volatile m_combined_write_stamp;

protected:
	mutable Recursive_mutex m_mutex;

public:
	Object_base()
		: m_auto_saves(false), m_combined_write_stamp(NULLPTR)
	{
		//
	}
	// 不要不写析构函数，否则 RTTI 将无法在动态库中使用。
	~Object_base();

public:
	bool is_auto_saving_enabled() const NOEXCEPT;
	void enable_auto_saving() const NOEXCEPT;
	void disable_auto_saving() const NOEXCEPT;

	bool invalidate() const NOEXCEPT;

	void *get_combined_write_stamp() const NOEXCEPT;
	void set_combined_write_stamp(void *stamp) const NOEXCEPT;

	virtual const char *get_table() const = 0;
	virtual void generate_sql(std::ostream &os) const = 0;
	virtual void fetch(const boost::shared_ptr<const Connection> &conn) = 0;
};

template<typename ValueT>
class Object_base::Field : NONCOPYABLE {
private:
	Object_base *const m_parent;
	ValueT m_value;

public:
	explicit Field(Object_base *parent, ValueT value = ValueT())
		: m_parent(parent), m_value(STD_MOVE_IDN(value))
	{
		//
	}

public:
	const ValueT &unlocked_get() const {
		return m_value;
	}
	ValueT get() const {
		const Recursive_mutex::Unique_lock lock(m_parent->m_mutex);
		return m_value;
	}
	void set(ValueT value, bool invalidates_parent = true){
		const Recursive_mutex::Unique_lock lock(m_parent->m_mutex);
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

extern template class Object_base::Field<bool>;
extern template class Object_base::Field<boost::int64_t>;
extern template class Object_base::Field<boost::uint64_t>;
extern template class Object_base::Field<double>;
extern template class Object_base::Field<std::string>;
extern template class Object_base::Field<Uuid>;
extern template class Object_base::Field<std::basic_string<unsigned char> >;

template<typename ValueT>
inline std::ostream &operator<<(std::ostream &os, const Object_base::Field<ValueT> &rhs){
	return os <<rhs.unlocked_get();
}
template<typename ValueT>
inline std::istream &operator>>(std::istream &is, Object_base::Field<ValueT> &rhs){
	ValueT value;
	if(is >>value){
		rhs.set(STD_MOVE(value));
	}
	return is;
}

class Object_base::Delimiter {
private:
	mutable std::size_t m_count;

public:
	Delimiter()
		: m_count(0)
	{
		//
	}

public:
	void apply(std::ostream &os) const {
		if(m_count != 0){
			os <<", ";
		}
		++m_count;
	}
};

inline std::ostream &operator<<(std::ostream &os, const Object_base::Delimiter &rhs){
	rhs.apply(os);
	return os;
}

extern void enqueue_for_saving(const boost::shared_ptr<Object_base> &obj);

template<typename ObjectT>
typename boost::enable_if_c<boost::is_base_of<Object_base, ObjectT>::value,
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
