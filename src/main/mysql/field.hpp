#ifndef POSEIDON_MYSQL_FIELD_HPP_
#define POSEIDON_MYSQL_FIELD_HPP_

#include "../../cxx_ver.hpp"
#include <string>
#include <sstream>
#include <vector>
#include <cstddef>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>

namespace Poseidon {

class MySqlFieldSnapshotBase : boost::noncopyable {
private:
	const char *const m_name;

protected:
	explicit MySqlFieldSnapshotBase(const char *name)
		: m_name(name)
	{
	}
	virtual ~MySqlFieldSnapshotBase(); // 定义在别处，否则 RTTI 会有问题。

public:
	const char *name() const {
		return m_name;
	}

	virtual void pack(unsigned int index, sql::PreparedStatement *ps) = 0;
};

template<typename T>
class MySqlFieldSnapshot : public MySqlFieldSnapshotBase {
private:
	T m_val;

public:
	MySqlFieldSnapshot(const char *name, T val)
		: MySqlFieldSnapshotBase(name), m_val(val)
	{
	}

public:
	T get() const {
		return m_val;
	}

	void pack(unsigned int index, sql::PreparedStatement *ps);
};

template<>
class MySqlFieldSnapshot<std::string> : public MySqlFieldSnapshotBase {
private:
	std::stringstream m_val;

public:
	MySqlFieldSnapshot(const char *name, const std::string &val)
		: MySqlFieldSnapshotBase(name), m_val(val)
	{
	}

public:
	std::string get() const {
		return m_val.str();
	}

	void pack(unsigned int index, sql::PreparedStatement *ps);
};

template class MySqlFieldSnapshot<bool>;
template class MySqlFieldSnapshot<signed char>;
template class MySqlFieldSnapshot<unsigned char>;
template class MySqlFieldSnapshot<short>;
template class MySqlFieldSnapshot<unsigned short>;
template class MySqlFieldSnapshot<int>;
template class MySqlFieldSnapshot<unsigned int>;
template class MySqlFieldSnapshot<long>;
template class MySqlFieldSnapshot<unsigned long>;
template class MySqlFieldSnapshot<long long>;
template class MySqlFieldSnapshot<unsigned long long>;
template class MySqlFieldSnapshot<float>;
template class MySqlFieldSnapshot<double>;
template class MySqlFieldSnapshot<long double>;
template class MySqlFieldSnapshot<std::string>;

class MySqlFieldBase : boost::noncopyable {
private:
	const char *const m_name;

	volatile unsigned long long m_timeStamp;

public:
	MySqlFieldBase(class MySqlObjectBase &owner, const char *name);
	virtual ~MySqlFieldBase(); // 定义在别处，否则 RTTI 会有问题。

public:
	const char *name() const {
		return m_name;
	}

	bool isInvalidated(unsigned long long time) const;
	void invalidate();

	virtual boost::shared_ptr<MySqlFieldSnapshotBase> snapshot() const = 0;
	virtual void fetch(unsigned int index, sql::ResultSet *rs) = 0;
};

template<typename T>
class MySqlField : public MySqlFieldBase {
private:
	T m_val;

public:
	MySqlField(MySqlObjectBase &owner, const char *name, T val = T())
		: MySqlFieldBase(owner, name), m_val(STD_MOVE(val))
	{
	}

public:
	const T &get() const {
		return m_val;
	}
	void set(T val){
		m_val = STD_MOVE(val);
		invalidate();
	}

	boost::shared_ptr<MySqlFieldSnapshotBase> snapshot() const {
		return boost::make_shared<MySqlFieldSnapshot<T> >(name(), get());
	}
	void fetch(unsigned int index, sql::ResultSet *rs);
};

template class MySqlField<bool>;
template class MySqlField<signed char>;
template class MySqlField<unsigned char>;
template class MySqlField<short>;
template class MySqlField<unsigned short>;
template class MySqlField<int>;
template class MySqlField<unsigned int>;
template class MySqlField<long>;
template class MySqlField<unsigned long>;
template class MySqlField<long long>;
template class MySqlField<unsigned long long>;
template class MySqlField<float>;
template class MySqlField<double>;
template class MySqlField<long double>;
template class MySqlField<std::string>;

}

#endif
