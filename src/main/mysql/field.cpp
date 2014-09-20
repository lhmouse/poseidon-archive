#include "../../precompiled.hpp"
#include "field.hpp"
#include "object_base.hpp"
#include "../atomic.hpp"
#include "../utilities.hpp"
using namespace Poseidon;

namespace Poseidon {

template<>
void MySqlField<bool>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setBoolean(index + 1, m_val);
}
template<>
void MySqlField<bool>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getBoolean(index + 1);
}

template<>
void MySqlField<signed char>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setInt(index + 1, m_val);
}
template<>
void MySqlField<signed char>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getInt(index + 1);
}

template<>
void MySqlField<unsigned char>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setUInt(index + 1, m_val);
}
template<>
void MySqlField<unsigned char>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getUInt(index + 1);
}

template<>
void MySqlField<short>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setInt(index + 1, m_val);
}
template<>
void MySqlField<short>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getInt(index + 1);
}

template<>
void MySqlField<unsigned short>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setUInt(index + 1, m_val);
}
template<>
void MySqlField<unsigned short>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getUInt(index + 1);
}

template<>
void MySqlField<int>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setInt(index + 1, m_val);
}
template<>
void MySqlField<int>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getInt(index + 1);
}

template<>
void MySqlField<unsigned int>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setUInt(index + 1, m_val);
}
template<>
void MySqlField<unsigned int>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getUInt(index + 1);
}

template<>
void MySqlField<long>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setInt64(index + 1, m_val);
}
template<>
void MySqlField<long>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getInt64(index + 1);
}

template<>
void MySqlField<unsigned long>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setUInt64(index + 1, m_val);
}
template<>
void MySqlField<unsigned long>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getUInt64(index + 1);
}

template<>
void MySqlField<long long>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setInt64(index + 1, m_val);
}
template<>
void MySqlField<long long>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getInt64(index + 1);
}

template<>
void MySqlField<unsigned long long>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setUInt64(index + 1, m_val);
}
template<>
void MySqlField<unsigned long long>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getUInt64(index + 1);
}

template<>
void MySqlField<float>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setDouble(index + 1, m_val);
}
template<>
void MySqlField<float>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getDouble(index + 1);
}

template<>
void MySqlField<double>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setDouble(index + 1, m_val);
}
template<>
void MySqlField<double>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getDouble(index + 1);
}

template<>
void MySqlField<long double>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &) const
{
	ps->setDouble(index + 1, m_val);
}
template<>
void MySqlField<long double>::fetch(unsigned int index, sql::ResultSet *rs){
	m_val = rs->getDouble(index + 1);
}

template<>
void MySqlField<std::string>::pack(unsigned int index, sql::PreparedStatement *ps,
	std::vector<boost::any> &contexts) const
{
	AUTO(ss, boost::make_shared<std::stringstream>(m_val));
	contexts.push_back(STD_MOVE(ss));
	ps->setBlob(index + 1, ss.get());
}
template<>
void MySqlField<std::string>::fetch(unsigned int index, sql::ResultSet *rs){
	rs->getString(index + 1)->swap(m_val); // 我能说脏话吗？
}

}

MySqlFieldBase::MySqlFieldBase(MySqlObjectBase &owner, const char *name)
	: m_name(name)
{
	owner.m_fields.push_back(boost::ref(*this));
}
MySqlFieldBase::~MySqlFieldBase(){
}
