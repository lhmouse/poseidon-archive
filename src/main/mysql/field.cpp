#include "../../precompiled.hpp"
#include "field.hpp"
#include "object_base.hpp"
#include "../atomic.hpp"
#include "../utilities.hpp"
using namespace Poseidon;

MySqlFieldSnapshotBase::~MySqlFieldSnapshotBase(){
}

MySqlFieldBase::MySqlFieldBase(MySqlObjectBase *owner, const char *name)
	: m_name(name)
{
	owner->m_fields.push_back(boost::ref(*this));
}
MySqlFieldBase::~MySqlFieldBase(){
}

bool MySqlFieldBase::isInvalidated(unsigned long long time) const {
	return atomicLoad(m_timeStamp) >= time;
}
void MySqlFieldBase::invalidate(){
	atomicStore(m_timeStamp, getMonoClock());
}

namespace Poseidon {

template<>
void MySqlFieldSnapshot<bool>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setBoolean(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<signed char>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setInt(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<unsigned char>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setUInt(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<short>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setInt(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<unsigned short>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setUInt(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<int>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setInt(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<unsigned int>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setUInt(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<long>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setInt64(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<unsigned long>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setUInt64(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<long long>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setInt64(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<unsigned long long>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setUInt64(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<float>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setDouble(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<double>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setDouble(index + 1, m_val);
}
template<>
void MySqlFieldSnapshot<long double>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setDouble(index + 1, m_val);
}
void MySqlFieldSnapshot<std::string>::pack(unsigned int index, sql::PreparedStatement *ps){
	ps->setBlob(index + 1, &m_val);
}

template<>
void MySqlField<bool>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getBoolean(index + 1);
}
template<>
void MySqlField<signed char>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getInt(index + 1);
}
template<>
void MySqlField<unsigned char>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getUInt(index + 1);
}
template<>
void MySqlField<short>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getInt(index + 1);
}
template<>
void MySqlField<unsigned short>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getUInt(index + 1);
}
template<>
void MySqlField<int>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getInt(index + 1);
}
template<>
void MySqlField<unsigned int>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getUInt(index + 1);
}
template<>
void MySqlField<long>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getInt64(index + 1);
}
template<>
void MySqlField<unsigned long>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getUInt64(index + 1);
}
template<>
void MySqlField<long long>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getInt64(index + 1);
}
template<>
void MySqlField<unsigned long long>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getUInt64(index + 1);
}
template<>
void MySqlField<float>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getDouble(index + 1);
}
template<>
void MySqlField<double>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getDouble(index + 1);
}
template<>
void MySqlField<long double>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getDouble(index + 1);
}
template<>
void MySqlField<std::string>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val.clear();

	std::istream *const is = rs->getBlob(index + 1);
	char temp[256];
	std::size_t count;
	while((count = is->readsome(temp, sizeof(temp))) != 0){
		m_val.append(temp, count);
	}
}

}
