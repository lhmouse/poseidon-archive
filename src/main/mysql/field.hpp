#ifndef POSEIDON_MYSQL_FIELD_HPP_
#define POSEIDON_MYSQL_FIELD_HPP_

#include "../../cxx_ver.hpp"
#include <string>
#include <sstream>
#include <vector>
#include <cstddef>
#include <boost/noncopyable.hpp>
#include <boost/any.hpp>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>

namespace Poseidon {

class MySqlFieldBase : boost::noncopyable {
private:
	const char *const m_name;

public:
	MySqlFieldBase(class MySqlObjectBase &owner, const char *name);
	virtual ~MySqlFieldBase(); // 定义在别处，否则 RTTI 会有问题。

public:
	const char *name() const {
		return m_name;
	}

	virtual void pack(unsigned int index, sql::PreparedStatement *ps,
		std::vector<boost::any> &contexts) const = 0;
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
	}

	void pack(unsigned int index, sql::PreparedStatement *ps,
		std::vector<boost::any> &contexts) const;
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
