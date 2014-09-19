#include "../../precompiled.hpp"
#include "field.hpp"
#include "../atomic.hpp"
#include "../utilities.hpp"
using namespace Poseidon;

MySqlFieldSnapshotBase::~MySqlFieldSnapshotBase(){
}

namespace Poseidon {

template<>
void MySqlFieldSnapshot<bool>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setBoolean(index, m_val);
}
template<>
void MySqlFieldSnapshot<bool>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getBoolean(index);
}

template<>
void MySqlFieldSnapshot<signed char>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt(index, m_val);
}
template<>
void MySqlFieldSnapshot<signed char>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt(index);
}

template<>
void MySqlFieldSnapshot<unsigned char>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setUInt(index, m_val);
}
template<>
void MySqlFieldSnapshot<unsigned char>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getUInt(index);
}

template<>
void MySqlFieldSnapshot<short>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt(index, m_val);
}
template<>
void MySqlFieldSnapshot<short>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt(index);
}

template<>
void MySqlFieldSnapshot<unsigned short>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setUInt(index, m_val);
}
template<>
void MySqlFieldSnapshot<unsigned short>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getUInt(index);
}

template<>
void MySqlFieldSnapshot<int>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt(index, m_val);
}
template<>
void MySqlFieldSnapshot<int>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt(index);
}

template<>
void MySqlFieldSnapshot<unsigned int>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setUInt(index, m_val);
}
template<>
void MySqlFieldSnapshot<unsigned int>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getUInt(index);
}

template<>
void MySqlFieldSnapshot<long>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt64(index, m_val);
}
template<>
void MySqlFieldSnapshot<long>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt64(index);
}

template<>
void MySqlFieldSnapshot<unsigned long>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt64(index, m_val);
}
template<>
void MySqlFieldSnapshot<unsigned long>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt64(index);
}

template<>
void MySqlFieldSnapshot<long long>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt64(index, m_val);
}
template<>
void MySqlFieldSnapshot<long long>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt64(index);
}

template<>
void MySqlFieldSnapshot<unsigned long long>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setInt64(index, m_val);
}
template<>
void MySqlFieldSnapshot<unsigned long long>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getInt64(index);
}

template<>
void MySqlFieldSnapshot<float>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setDouble(index, m_val);
}
template<>
void MySqlFieldSnapshot<float>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getDouble(index);
}

template<>
void MySqlFieldSnapshot<double>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setDouble(index, m_val);
}
template<>
void MySqlFieldSnapshot<double>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getDouble(index);
}

template<>
void MySqlFieldSnapshot<long double>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setDouble(index, m_val);
}
template<>
void MySqlFieldSnapshot<long double>::deserialize(
	unsigned int index, sql::ResultSet *rs)
{
	m_val = rs->getDouble(index);
}

void MySqlFieldSnapshot<std::string>::serialize(
	unsigned int index, sql::PreparedStatement *ps)
{
	ps->setBlob(index, &m_val);
}
void MySqlFieldSnapshot<std::string>::deserialize(
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

}

MySqlFieldBase::~MySqlFieldBase(){
}

void MySqlFieldBase::invalidate(){
	atomicStore(m_timeStamp, getMonoClock());
}

bool MySqlFieldBase::isInvalidated(unsigned long long time) const {
	return atomicLoad(m_timeStamp) >= time;
}
