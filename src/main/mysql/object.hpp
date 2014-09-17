#ifndef POSEIDON_MYSQL_OBJECT_HPP_
#define POSEIDON_MYSQL_OBJECT_HPP_

#include "../../cxx_ver.hpp"
#include "../virtual_shared_from_this.hpp"
#include <map>
#include <string>

namespace Poseidon {

class MySqlObject;

template<typename ValueT>
class MySqlField : public virtual VirtualSharedFromThis {
private:
	ValueT m_val;

public:
	explicit MySqlField(MySqlObject *owner, ValueT val = ValueT())
		: m_val(STD_MOVE(val))
	{
	}

public:
	const ValueT &get() const {
		return m_val;
	}
	void set(ValueT val){
		using std::swap;
		swap(val, m_val);
	}
};

class MySqlObject : public virtual VirtualSharedFromThis {
private:
	
};

}

#endif
