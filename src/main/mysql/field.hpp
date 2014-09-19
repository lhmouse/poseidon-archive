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
protected:
	virtual ~MySqlFieldSnapshotBase(); // 定义在别处，否则 RTTI 会有问题。

public:
	virtual void serialize(unsigned int index, sql::PreparedStatement *ps) = 0;
	virtual void deserialize(unsigned int index, sql::ResultSet *rs) = 0;
};

template<typename T>
class MySqlFieldSnapshot : public MySqlFieldSnapshotBase {
private:
	T m_val;

public:
	explicit MySqlFieldSnapshot(T val)
		: m_val(val)
	{
	}

public:
	const T &get() const {
		return m_val;
	}

	void serialize(unsigned int index, sql::PreparedStatement *ps);
	void deserialize(unsigned int index, sql::ResultSet *rs);
};

template<>
class MySqlFieldSnapshot<std::string> : public MySqlFieldSnapshotBase {
private:
	std::stringstream m_val;

public:
	explicit MySqlFieldSnapshot(const std::string &val)
		: m_val(val)
	{
	}

public:
	std::string get() const {
		return m_val.str();
	}

	void serialize(unsigned int index, sql::PreparedStatement *ps);
	void deserialize(unsigned int index, sql::ResultSet *rs);
};

template<>
inline void MySqlFieldSnapshot<bool>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setBoolean(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<bool>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getBoolean(index);
}

template<>
inline void MySqlFieldSnapshot<signed char>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<signed char>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt(index);
}

template<>
inline void MySqlFieldSnapshot<unsigned char>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setUInt(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<unsigned char>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getUInt(index);
}

template<>
inline void MySqlFieldSnapshot<short>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<short>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt(index);
}

template<>
inline void MySqlFieldSnapshot<unsigned short>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setUInt(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<unsigned short>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getUInt(index);
}

template<>
inline void MySqlFieldSnapshot<int>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<int>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt(index);
}

template<>
inline void MySqlFieldSnapshot<unsigned int>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setUInt(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<unsigned int>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getUInt(index);
}

template<>
inline void MySqlFieldSnapshot<long>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt64(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<long>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt64(index);
}

template<>
inline void MySqlFieldSnapshot<unsigned long>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt64(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<unsigned long>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt64(index);
}

template<>
inline void MySqlFieldSnapshot<long long>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt64(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<long long>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt64(index);
}

template<>
inline void MySqlFieldSnapshot<unsigned long long>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt64(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<unsigned long long>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt64(index);
}

template<>
inline void MySqlFieldSnapshot<float>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setDouble(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<float>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getDouble(index);
}

template<>
inline void MySqlFieldSnapshot<double>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setDouble(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<double>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getDouble(index);
}

template<>
inline void MySqlFieldSnapshot<long double>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setDouble(index, m_val);
}
template<>
inline void MySqlFieldSnapshot<long double>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getDouble(index);
}

inline void MySqlFieldSnapshot<std::string>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setBlob(index, &m_val);
}
inline void MySqlFieldSnapshot<std::string>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val.str(std::string());

	std::istream *const is = rs->getBlob(index);
	for(;;){
		char temp[1024];
		std::size_t count = is->readsome(temp, sizeof(temp));
		if(count == 0){
			break;
		}
		m_val.write(temp, count);
	}
}

class MySqlFieldBase : boost::noncopyable {
private:
	const char *const m_name;

	bool m_invalidated;

public:
	explicit MySqlFieldBase(const char *name)
		: m_name(name)
	{
	}
	virtual ~MySqlFieldBase(); // 定义在别处，否则 RTTI 会有问题。

public:
	const char *name() const {
		return m_name;
	}

	virtual boost::shared_ptr<MySqlFieldSnapshotBase> snapshot() const = 0;
	virtual void fetch(const boost::shared_ptr<MySqlFieldSnapshotBase> &snapshot) = 0;
};

template<typename T>
class MySqlField : public MySqlFieldBase {
private:
	T m_val;

public:
	explicit MySqlField(const char *name, T val = T())
		: MySqlFieldBase(name), m_val(STD_MOVE(val))
	{
	}

public:
	const T &get() const {
		return m_val;
	}
	void set(T val){
		m_val = STD_MOVE(val);
	}

	boost::shared_ptr<MySqlFieldSnapshotBase> snapshot() const {
		return boost::make_shared<MySqlFieldSnapshot<T> >(m_val);
	}
	void fetch(const boost::shared_ptr<MySqlFieldSnapshotBase> &snapshot){
		m_val = static_cast<const MySqlFieldSnapshot<T> *>(snapshot.get())->get();
	}
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
